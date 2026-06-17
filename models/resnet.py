import torch
import torch.nn as nn
import torch.nn.functional as F

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)

        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, h, w = x.shape

        y = self.avgpool(x)      # (B,C,1,1)
        y = y.view(b, c)         # (B,C)
        y = self.fc(y)           # (B,C)
        y = y.view(b, c, 1, 1)   # (B,C,1,1)

        return x * y

class ChannelAttention(nn.Module):
    def __init__(self, channels, reduction=16):
        super(ChannelAttention, self).__init__()

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.maxpool = nn.AdaptiveMaxPool2d(1)

        self.mlp = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, kernel_size=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels // reduction, channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.mlp(self.avgpool(x))
        max_out = self.mlp(self.maxpool(x))

        weight = self.sigmoid(avg_out + max_out)

        return x * weight
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        padding = kernel_size // 2

        self.conv = nn.Conv2d(
            2,
            1,
            kernel_size=kernel_size,
            padding=padding,
            bias=False
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)

        attn = torch.cat([avg_out, max_out], dim=1)
        weight = self.sigmoid(self.conv(attn))

        return x * weight
    
class CBAMBlock(nn.Module):
    def __init__(self, channels, reduction=16, spatial_kernel=7):
        super(CBAMBlock, self).__init__()

        self.channel_attention = ChannelAttention(
            channels,
            reduction=reduction
        )

        self.spatial_attention = SpatialAttention(
            kernel_size=spatial_kernel
        )

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)

        return x

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(
            in_channels, out_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = nn.Conv2d(
            out_channels, out_channels,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(out_channels)

        # self.se = SEBlock(out_channels)
        self.attn = CBAMBlock(out_channels)

        self.relu = nn.ReLU(inplace=True)

        #shortcut分支，负责把identity的shape对其到out
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        # out = self.se(out)
        out = self.attn(out)
        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class ResNet18(nn.Module):
    def __init__(self, num_classes=2):
        super(ResNet18, self).__init__()

        self.in_channels = 64

        self.conv1 = nn.Conv2d(
            3, 64,
            kernel_size=3,
            stride=1,
            padding=1,
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        # self.maxpool = nn.MaxPool2d(
        #     kernel_size=3,
        #     stride=2,
        #     padding=1
        # )
        self.maxpool = nn.Identity()

        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def _make_layer(self, out_channels, blocks, stride):
        layers = []

        layers.append(
            BasicBlock(
                self.in_channels,
                out_channels,
                stride=stride
            )
        )

        self.in_channels = out_channels

        for _ in range(1, blocks):
            layers.append(
                BasicBlock(
                    self.in_channels,
                    out_channels,
                    stride=1
                )
            )

        return nn.Sequential(*layers)
    
    def forward_features(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x1 = self.layer1(x)  # 1/4
        x2 = self.layer2(x1) # 1/8
        x3 = self.layer3(x2) # 1/16
        x4 = self.layer4(x3) # 1/32

        return x1, x2, x3, x4

    def forward(self, x):
        #print("input:", x.shape)

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        #print("after conv1:", x.shape)

        x = self.maxpool(x)
        #print("after maxpool:", x.shape)

        x = self.layer1(x)
        #print("after layer1:", x.shape)

        x = self.layer2(x)
        #print("after layer2:", x.shape)

        x = self.layer3(x)
        #  print("after layer3:", x.shape)

        x = self.layer4(x)
        #print("after layer4:", x.shape)

        x = self.avgpool(x)
        #print("after avgpool:", x.shape)

        x = torch.flatten(x, 1)
        #print("after flatten:", x.shape)

        x = self.fc(x)
        #print("after fc:", x.shape)

        return x

# if __name__ == "__main__":
#     x = torch.randn(1, 3, 224, 224)
#     model = ResNet18(num_classes=2)
#     block = BasicBlock(64, 64)
#     x1, x2, x3, x4 = model.forward_features(x)

#     print("x1:", x1.shape)
#     print("x2:", x2.shape)
#     print("x3:", x3.shape)
#     print("x4:", x4.shape)

if __name__ == "__main__":
    x = torch.randn(1, 64, 56, 56)
    block = BasicBlock(64, 64)
    y = block(x)
    print(y.shape)

    x0 = torch.randn(1, 3, 224, 224)
    model = ResNet18(num_classes=2)
    x1, x2, x3, x4 = model.forward_features(x0)
    print("x1:", x1.shape)
    print("x2:", x2.shape)
    print("x3:", x3.shape)
    print("x4:", x4.shape)
