import torch
import torch.nn as nn

from models.resnet import ResNet18, BasicBlock
from models.light_ssm_v1 import LightSSM


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)

        self.reduce = nn.Sequential(
            nn.Conv2d(in_channels + skip_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

        self.res = BasicBlock(out_channels, out_channels)

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.reduce(x)
        return self.res(x)


class ResUNetLightSSM(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.encoder = ResNet18(num_classes=2)

        # ⭐只在decoder用SSM（关键设计）
        self.ssm3 = LightSSM(256)
        self.ssm2 = LightSSM(128)

        self.up3 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 64)

        self.refine = LightSSM(64)

        self.out_conv = nn.Conv2d(64, num_classes, 1)

    def forward(self, x):
        size = x.shape[-2:]

        x1, x2, x3, x4 = self.encoder.forward_features(x)

        x3 = self.ssm3(x3)
        x2 = self.ssm2(x2)

        d3 = self.up3(x4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)

        d1 = self.refine(d1)

        out = nn.functional.interpolate(d1, size=size, mode="bilinear", align_corners=True)
        out = self.out_conv(out)

        return out