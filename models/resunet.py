import torch
import torch.nn as nn

from models.resnet import ResNet18


class UpBlock(nn.Module):
    def __init__(self, in_channels, skip_channels, out_channels):
        super().__init__()

        self.up = nn.Upsample(
            scale_factor=2,
            mode="bilinear",
            align_corners=True
        )

        self.conv = nn.Sequential(
            nn.Conv2d(
                in_channels + skip_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x, skip):
        x = self.up(x)
        x = torch.cat([x, skip], dim=1)
        x = self.conv(x)
        return x


class ResUNet(nn.Module):
    def __init__(self, num_classes=1):
        super().__init__()

        self.encoder = ResNet18(num_classes=2)

        self.up3 = UpBlock(512, 256, 256)
        self.up2 = UpBlock(256, 128, 128)
        self.up1 = UpBlock(128, 64, 64)

        # self.final_up = nn.Upsample(
        #     scale_factor=4,
        #     mode="bilinear",
        #     align_corners=True
        # )
        self.final_up = nn.Identity()

        self.out_conv = nn.Conv2d(
            64,
            num_classes,
            kernel_size=1
        )

    def forward(self, x):
        input_size = x.shape[-2:]

        x1, x2, x3, x4 = self.encoder.forward_features(x)

        d3 = self.up3(x4, x3)
        d2 = self.up2(d3, x2)
        d1 = self.up1(d2, x1)

        out = self.final_up(d1)

        if out.shape[-2:] != input_size:
            out = nn.functional.interpolate(
                out,
                size=input_size,
                mode="bilinear",
                align_corners=True
            )

        out = self.out_conv(out)

        return out


if __name__ == "__main__":
    x = torch.randn(1, 3, 256, 256)

    model = ResUNet(num_classes=1)

    y = model(x)

    # print("input:", x.shape)
    # print("output:", y.shape)