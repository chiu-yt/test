import math

import torch
import torch.nn.functional as F
from torch import nn


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
        self.group_norm_groups = self._pick_group_norm_groups(int(model_cfg.get('GROUP_NORM_GROUPS', 8)), hidden_channels)
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
            torch.tensor(float(model_cfg.get('RESIDUAL_SCALE_INIT', 0.0)), dtype=torch.float32)
        )
        if point_cloud_range is not None:
            self.register_buffer('point_cloud_range', torch.tensor(point_cloud_range, dtype=torch.float32))
        else:
            self.register_buffer('point_cloud_range', torch.zeros(6, dtype=torch.float32))

        self._init_identity()

    def _init_identity(self):
        for module in [self.shared_refine, self.shared_gate, self.proposal_geom_proj, self.proposal_shared_head, self.proposal_router]:
            self._zero_last_affine(module)
        for module in self.class_shared_heads.values():
            self._zero_last_affine(module)
        for module in self.class_gate_heads.values():
            self._zero_last_affine(module)

    @staticmethod
    def _pick_group_norm_groups(requested_groups, channels):
        max_groups = max(1, min(int(requested_groups), int(channels)))
        for groups in range(max_groups, 0, -1):
            if channels % groups == 0:
                return groups
        return 1

    @staticmethod
    def _zero_last_affine(module):
        if isinstance(module, nn.Sequential):
            for submodule in reversed(list(module)):
                if isinstance(submodule, (nn.Linear, nn.Conv2d)):
                    nn.init.zeros_(submodule.weight)
                    nn.init.zeros_(submodule.bias)
                    break

    def _has_valid_range(self):
        return bool(torch.any(self.point_cloud_range != 0))

    def _world_to_grid(self, centers, height, width):
        x_min, y_min, _, x_max, y_max, _ = self.point_cloud_range.tolist()
        x_norm = (centers[:, 0] - x_min) / max(x_max - x_min, 1e-6) * 2.0 - 1.0
        y_norm = (centers[:, 1] - y_min) / max(y_max - y_min, 1e-6) * 2.0 - 1.0
        return torch.stack([x_norm, y_norm], dim=-1)

    def _sample_center_tokens(self, bev_feat, centers):
        height, width = bev_feat.shape[-2:]
        grid = self._world_to_grid(centers, height, width).view(1, -1, 1, 2)
        sampled = F.grid_sample(bev_feat, grid, mode='bilinear', align_corners=True)
        return sampled.squeeze(-1).squeeze(0).transpose(0, 1).contiguous()

    def _proposal_geometry(self, boxes, height, width):
        x_min, y_min, z_min, x_max, y_max, z_max = self.point_cloud_range.tolist()
        center = boxes[:, :3]
        dims = boxes[:, 3:6]
        yaw = boxes[:, 6]
        score = boxes[:, 8]

        x_norm = (center[:, 0] - x_min) / max(x_max - x_min, 1e-6)
        y_norm = (center[:, 1] - y_min) / max(y_max - y_min, 1e-6)
        z_norm = (center[:, 2] - z_min) / max(z_max - z_min, 1e-6)

        geom = torch.stack([
            x_norm,
            y_norm,
            z_norm,
            dims[:, 0] / max(x_max - x_min, 1e-6),
            dims[:, 1] / max(y_max - y_min, 1e-6),
            dims[:, 2] / max(z_max - z_min, 1e-6),
            torch.sin(yaw),
            score,
        ], dim=-1)
        return geom

    def _stamp_residual(self, residual_map, center_xy, residual_vec, gate, height, width):
        radius = max(self.pool_size, 1)
        cx = int(torch.round(center_xy[0]).item())
        cy = int(torch.round(center_xy[1]).item())
        x0 = max(0, cx - radius)
        x1 = min(width, cx + radius + 1)
        y0 = max(0, cy - radius)
        y1 = min(height, cy + radius + 1)
        if x0 >= x1 or y0 >= y1:
            return

        yy = torch.arange(y0, y1, device=residual_map.device, dtype=residual_map.dtype)
        xx = torch.arange(x0, x1, device=residual_map.device, dtype=residual_map.dtype)
        yy, xx = torch.meshgrid(yy, xx, indexing='ij')
        dist2 = (yy - float(cy)) ** 2 + (xx - float(cx)) ** 2
        sigma = float(radius) / 2.0 if radius > 1 else 1.0
        kernel = torch.exp(-dist2 / max(2.0 * sigma * sigma, 1e-6))
        residual_map[:, y0:y1, x0:x1] += gate * residual_vec.view(-1, 1, 1) * kernel.view(1, y1 - y0, x1 - x0)

    def _proposal_residual_map(self, shared_context, proposal_boxes, proposal_mask):
        batch_size, _, height, width = shared_context.shape
        residual_map = shared_context.new_zeros((batch_size, self.channels, height, width))
        if proposal_boxes is None or proposal_mask is None:
            return residual_map

        for b_idx in range(batch_size):
            cur_mask = proposal_mask[b_idx]
            if cur_mask.numel() == 0 or not bool(cur_mask.any()):
                continue

            cur_boxes = proposal_boxes[b_idx][cur_mask]
            if cur_boxes.numel() == 0:
                continue

            if cur_boxes.shape[0] < self.min_active_proposals:
                continue

            cls_ids = cur_boxes[:, 7].round().long()
            scores = cur_boxes[:, 8]
            valid = (cls_ids > 0) & torch.isfinite(scores) & (scores >= self.min_proposal_score)
            if len(self.target_class_ids) > 0:
                class_valid = torch.zeros_like(valid)
                for class_id in self.target_class_ids:
                    class_valid |= cls_ids == class_id
                valid &= class_valid
            if not bool(valid.any()):
                continue

            cur_boxes = cur_boxes[valid]
            cls_ids = cls_ids[valid]
            scores = scores[valid]

            if self.topk_per_class > 0:
                filtered_boxes = []
                filtered_cls = []
                filtered_scores = []
                for class_id in self.target_class_ids:
                    class_mask = cls_ids == class_id
                    if not bool(class_mask.any()):
                        continue
                    class_boxes = cur_boxes[class_mask]
                    class_scores = scores[class_mask]
                    order = torch.argsort(class_scores, descending=True)
                    class_boxes = class_boxes[order[: self.topk_per_class]]
                    class_scores = class_scores[order[: self.topk_per_class]]
                    filtered_boxes.append(class_boxes)
                    filtered_cls.append(torch.full((class_boxes.shape[0],), class_id, device=class_boxes.device, dtype=torch.long))
                    filtered_scores.append(class_scores)
                if len(filtered_boxes) == 0:
                    continue
                cur_boxes = torch.cat(filtered_boxes, dim=0)
                cls_ids = torch.cat(filtered_cls, dim=0)
                scores = torch.cat(filtered_scores, dim=0)

            centers = cur_boxes[:, :3]
            center_tokens = self._sample_center_tokens(shared_context[b_idx:b_idx + 1], centers)
            geom_tokens = self.proposal_geom_proj(self._proposal_geometry(cur_boxes, height, width))
            class_tokens = self.class_embeddings(cls_ids.clamp(min=0, max=self.class_embeddings.num_embeddings - 1))
            proposal_tokens = center_tokens + geom_tokens + class_tokens

            for i in range(cur_boxes.shape[0]):
                cls_id = int(cls_ids[i].item())
                score = float(scores[i].item())
                class_key = str(cls_id)
                shared_delta = self.proposal_shared_head(proposal_tokens[i])
                class_delta = self.class_shared_heads[class_key](proposal_tokens[i]) if class_key in self.class_shared_heads else 0.0
                if self.router_mode == 'shared_only':
                    delta_vec = shared_delta
                elif self.router_mode == 'class_specific':
                    delta_vec = class_delta
                else:
                    delta_vec = shared_delta + class_delta

                route = self.proposal_router(proposal_tokens[i])
                if class_key in self.class_gate_heads:
                    route = route + self.class_gate_heads[class_key](proposal_tokens[i])
                gate = torch.sigmoid(route).view(1) * score

                center_xy = self._world_to_grid(cur_boxes[i:i + 1, :3], height, width)[0]
                center_xy = torch.stack([
                    (center_xy[0] + 1.0) * 0.5 * (width - 1),
                    (center_xy[1] + 1.0) * 0.5 * (height - 1),
                ])
                self._stamp_residual(residual_map[b_idx], center_xy, delta_vec, gate, height, width)

        return residual_map

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
            proposal_residual = self._proposal_residual_map(shared_context, proposal_boxes, proposal_mask)

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
