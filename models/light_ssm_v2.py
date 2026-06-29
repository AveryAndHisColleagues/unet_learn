import torch
import torch.nn as nn


class LightSSM(nn.Module):
    def __init__(self, dim):
        super().__init__()

        self.local = nn.Sequential(
            nn.Conv2d(dim, dim, 3, padding=1, groups=dim),
            nn.Conv2d(dim, dim, 1),
            nn.BatchNorm2d(dim),
            nn.GELU()
        )

        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dim, dim // 4, 1),
            nn.GELU(),
            nn.Conv2d(dim // 4, dim, 1),
            nn.Sigmoid()
        )

        self.h = nn.Conv2d(dim, dim, (1, 3), padding=(0, 1), groups=dim)
        self.w = nn.Conv2d(dim, dim, (3, 1), padding=(1, 0), groups=dim)

        self.fuse = nn.Conv2d(dim * 3, dim, 1)

    def forward(self, x):
        residual = x

        x1 = self.local(x)
        x1 = x1 * self.gate(x1)

        h = self.h(x1)
        w = self.w(x1)

        out = torch.cat([x1, h, w], dim=1)
        out = self.fuse(out)

        return out + residual