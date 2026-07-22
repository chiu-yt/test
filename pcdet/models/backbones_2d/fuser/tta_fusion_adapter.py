import torch
from torch import nn


class BEVFusionTTAAdapter(nn.Module):
    def __init__(self, model_cfg, input_channels=None):
        super().__init__()
        self.model_cfg = model_cfg
        channels = int(model_cfg.get('IN_CHANNEL', input_channels))
        image_channels = int(model_cfg.get('IMAGE_CHANNEL', 80))
        lidar_channels = int(model_cfg.get('LIDAR_CHANNEL', 256))
        hidden_channels = int(model_cfg.get('HIDDEN_CHANNEL', max(channels // 4, 32)))

        self.fused_proj = nn.Conv2d(channels, hidden_channels, kernel_size=1, bias=False)
        self.image_proj = nn.Conv2d(image_channels, hidden_channels, kernel_size=1, bias=False)
        self.lidar_proj = nn.Conv2d(lidar_channels, hidden_channels, kernel_size=1, bias=False)

        self.refine = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
        )
        self.gate = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(hidden_channels),
            nn.ReLU(True),
            nn.Conv2d(hidden_channels, channels, kernel_size=1, bias=True),
            nn.Sigmoid(),
        )
        self.residual_scale = nn.Parameter(
            torch.tensor(float(model_cfg.get('RESIDUAL_SCALE_INIT', 0.0)), dtype=torch.float32)
        )
        self._init_identity()

    def _init_identity(self):
        nn.init.zeros_(self.refine[-1].weight)
        nn.init.zeros_(self.refine[-1].bias)
        nn.init.zeros_(self.gate[-2].weight)
        nn.init.zeros_(self.gate[-2].bias)

    def forward(self, batch_dict):
        fused_bev = batch_dict['spatial_features']
        adapter_context = self.fused_proj(fused_bev)

        image_bev = batch_dict.get('spatial_features_img', None)
        if image_bev is not None:
            adapter_context = adapter_context + self.image_proj(image_bev)

        lidar_bev = batch_dict.get('spatial_features_lidar', None)
        if lidar_bev is not None:
            adapter_context = adapter_context + self.lidar_proj(lidar_bev)

        residual = self.refine(adapter_context)
        gate = self.gate(adapter_context)
        batch_dict['spatial_features'] = fused_bev + self.residual_scale.type_as(residual) * gate * residual
        batch_dict['tta_adapter_gate_mean'] = gate.detach().mean()
        return batch_dict
