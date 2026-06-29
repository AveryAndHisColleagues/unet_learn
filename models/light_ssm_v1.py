import torch
import torch.nn as nn
import torch.nn.functional as F


class LightSSM(nn.Module):
    """
    Lightweight SSM for segmentation
    (no loop, no GRU, fully parallel)
    """

    def __init__(self, dim):
        super().__init__()

        # 1. local feature extraction
        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim, bias=False),
            nn.Conv2d(dim, dim, 1, bias=False),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        # 2. "state mixing" (channel interaction)
        self.mixer = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        # 3. direction-aware scan (H + W)
        self.h_scan = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        self.w_scan = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)

        self.fuse = nn.Conv2d(dim * 3, dim, 1)

    def forward(self, x):
        residual = x

        # local spatial modeling
        local = self.local(x)

        # channel-wise gating (state-like behavior)
        gate = self.mixer(local)
        gated = local * gate

        # directional context (like simplified SSM scan)
        h = self.h_scan(gated)
        w = self.w_scan(gated)

        # fuse
        out = torch.cat([local, h, w], dim=1)
        out = self.fuse(out)

        return out + residual