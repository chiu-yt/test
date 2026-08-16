import torch
import torch.nn.functional as F
from torch import nn

from pcdet.models.backbones_2d.fuser.tta_fusion_adapter_utils import pick_group_norm_groups, proposal_residual_map, zero_last_affine


class BEVFusionTTAAdapter(nn.Module):
    def __init__(self, model_cfg, input_channels=None, point_cloud_range=None, class_names=None):
        super().__init__()
        self.model_cfg = model_cfg
        channels = int(model_cfg.get('IN_CHANNEL', input_channels))
        self.channels = channels
        image_channels = int(model_cfg.get('IMAGE_CHANNEL', 80))
        lidar_channels = int(model_cfg.get('LIDAR_CHANNEL', 256))
        hidden_channels = int(model_cfg.get('HIDDEN_CHANNEL', max(channels // 4, 32)))

        self.use_proposal_context = bool(model_cfg.get('USE_PROPOSAL_CONTEXT', False))
        self.min_proposal_score = float(model_cfg.get('MIN_PROPOSAL_SCORE', 0.0))
        self.min_active_proposals = int(model_cfg.get('MIN_ACTIVE_PROPOSALS', 1))
        self.topk_per_class = int(model_cfg.get('TOPK_PER_CLASS', 0))
        self.router_mode = str(model_cfg.get('ROUTER_MODE', 'shared_plus_class')).lower()
        self.pool_size = int(model_cfg.get('POOL_SIZE', 3))
        self.group_norm_groups = pick_group_norm_groups(int(model_cfg.get('GROUP_NORM_GROUPS', 8)), hidden_channels)
        density_cfg = model_cfg.get('SG_DFA', None)
        self.sg_dfa_enabled = bool(density_cfg is not None and density_cfg.get('ENABLED', False))

        self.fused_proj = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.image_proj = nn.Conv2d(image_channels, hidden_channels, kernel_size=1, bias=False)
        self.lidar_proj = nn.Conv2d(lidar_channels, hidden_channels, kernel_size=1, bias=False)
        if self.sg_dfa_enabled:
            self.density_proj = nn.Conv2d(1, hidden_channels, kernel_size=1, bias=False)

        self.shared_refine = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self.group_norm_groups, hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
        )
        self.shared_gate = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(self.group_norm_groups, hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )

        self.proposal_geom_proj = nn.Sequential(
            nn.Linear(8, hidden_channels),
            nn.ReLU(True),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.proposal_shared_head = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(True),
            nn.Linear(hidden_channels, channels),
        )
        self.proposal_router = nn.Sequential(
            nn.Linear(hidden_channels, hidden_channels),
            nn.ReLU(True),
            nn.Linear(hidden_channels, 1),
        )

        self.class_names = list(class_names or [])
        self.class_name_to_id = {name: idx + 1 for idx, name in enumerate(self.class_names)}
        self.target_class_names = list(model_cfg.get('TARGET_CLASSES', []))
        self.target_class_ids = [self.class_name_to_id[name] for name in self.target_class_names if name in self.class_name_to_id]
        self.class_embeddings = nn.Embedding(max(len(self.class_names), 1) + 1, hidden_channels)
        self.class_shared_heads = nn.ModuleDict()
        self.class_gate_heads = nn.ModuleDict()
        for class_id in self.target_class_ids:
            key = str(class_id)
            self.class_shared_heads[key] = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(True),
                nn.Linear(hidden_channels, channels),
            )
            self.class_gate_heads[key] = nn.Sequential(
                nn.Linear(hidden_channels, hidden_channels),
                nn.ReLU(True),
                nn.Linear(hidden_channels, 1),
            )

        self.residual_scale = nn.Parameter(
            torch.tensor(float(model_cfg.get('RESIDUAL_SCALE_INIT', 0.1)), dtype=torch.float32)
        )
        if point_cloud_range is not None:
            self.register_buffer('point_cloud_range', torch.tensor(point_cloud_range, dtype=torch.float32))
        else:
            self.register_buffer('point_cloud_range', torch.zeros(6, dtype=torch.float32))

        self._init_identity()

    def _init_identity(self):
        for module in [self.shared_refine, self.shared_gate, self.proposal_geom_proj, self.proposal_shared_head, self.proposal_router]:
            zero_last_affine(module)
        for module in self.class_shared_heads.values():
            zero_last_affine(module)
        for module in self.class_gate_heads.values():
            zero_last_affine(module)

    def _has_valid_range(self):
        return bool(torch.any(self.point_cloud_range != 0))

    def forward(self, batch_dict):
        fused_bev = batch_dict['spatial_features']
        shared_context = self.fused_proj(fused_bev)

        image_bev = batch_dict.get('spatial_features_img', None)
        if image_bev is not None:
            shared_context = shared_context + self.image_proj(image_bev)

        lidar_bev = batch_dict.get('spatial_features_lidar', None)
        if lidar_bev is not None:
            shared_context = shared_context + self.lidar_proj(lidar_bev)

        density_map = batch_dict.get('tta_density_map', None)
        density_mean = fused_bev.new_zeros(())
        density_nonzero = fused_bev.new_zeros(())
        if self.sg_dfa_enabled and density_map is not None:
            density_map = F.interpolate(
                density_map.float(),
                size=shared_context.shape[-2:],
                mode='bilinear',
                align_corners=False,
            )
            shared_context = shared_context + self.density_proj(density_map)
            density_gate = 0.5 + 0.5 * torch.sigmoid(density_map)
            density_mean = density_map.detach().mean()
            density_nonzero = (density_map.detach() > 0).float().mean()
        else:
            density_gate = 1.0

        shared_residual = self.shared_refine(shared_context)
        shared_gate = self.shared_gate(shared_context) * density_gate

        proposal_boxes = batch_dict.get('tta_proposal_boxes', None)
        proposal_mask = batch_dict.get('tta_proposal_mask', None)
        proposal_residual = fused_bev.new_zeros(fused_bev.shape)
        if self.use_proposal_context and proposal_boxes is not None and proposal_mask is not None and self._has_valid_range():
            proposal_residual = proposal_residual_map(self, shared_context, proposal_boxes, proposal_mask)

        batch_dict['spatial_features'] = fused_bev + self.residual_scale.type_as(shared_residual) * (
            shared_gate * shared_residual + proposal_residual
        )
        batch_dict['tta_adapter_gate_mean'] = shared_gate.detach().mean()
        batch_dict['tta_adapter_density_mean'] = density_mean
        batch_dict['tta_adapter_density_nonzero'] = density_nonzero
        batch_dict['tta_adapter_residual_mean'] = (
            shared_residual.detach().abs().mean() + proposal_residual.detach().abs().mean()
        )
        batch_dict['tta_adapter_residual_scale'] = self.residual_scale.detach()
        batch_dict['tta_adapter_output_input_rel_diff'] = (
            (batch_dict['spatial_features'].detach() - fused_bev.detach()).abs().mean()
            / fused_bev.detach().abs().mean().clamp(min=1e-6)
        )
        return batch_dict
