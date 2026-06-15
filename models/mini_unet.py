import torch
import torch.nn as nn
import torch.nn.functional as F

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),

            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels), 
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.net(x)
    
class Down(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.net = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )

    def forward(self, x):
        return self.net(x)
    
# class Up(nn.Module):
#     def __init__(self, in_channels, out_channels):
#         super().__init__()

#         self.up = nn.ConvTranspose2d(
#             in_channels,
#             in_channels // 2,
#             kernel_size=2,
#             stride=2
#         )

#         self.conv = DoubleConv(in_channels, out_channels)

#     def forward(self, x_decoder, x_encoder):
#         x_decoder = self.up(x_decoder)

#         x = torch.cat([x_encoder, x_decoder], dim=1)

#         return self.conv(x)

class Up(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()

        self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)  # 通道数变为 out_channels
        self.conv = DoubleConv(out_channels * 2, out_channels)  # 输入通道是上采样结果 + 编码器特征图

    def forward(self, x_decoder, x_encoder):
        x_decoder = self.up(x_decoder)
        # 这里需要处理尺寸的微小差异，可以用中心裁剪
        diffY = x_encoder.size()[2] - x_decoder.size()[2]
        diffX = x_encoder.size()[3] - x_decoder.size()[3]
        x_decoder = F.pad(x_decoder, [diffX // 2, diffX - diffX // 2,
                                      diffY // 2, diffY - diffY // 2])
        x = torch.cat([x_encoder, x_decoder], dim=1)
        return self.conv(x)
    
class MiniUNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=1): # num_classes=1 类别数为1时，表示二分类问题，输出一个通道的概率图；num_classes>1时，表示多分类问题，输出num_classes个通道的概率图
        super().__init__()

        self.inc = DoubleConv(in_channels, 64)

        self.down1 = Down(64, 128)
        self.down2 = Down(128, 256)

        self.bottleneck = Down(256, 512)

        self.up1 = Up(512, 256)
        self.up2 = Up(256, 128)
        self.up3 = Up(128, 64)

        self.out = nn.Conv2d(64, num_classes, kernel_size=1)

    def forward(self, x):
        # x1 = self.inc(x)
        # x2 = self.down1(x1)
        # x3 = self.down2(x2)
        # x4 = self.bottleneck(x3)

        # x = self.up1(x4, x3)
        # x = self.up2(x, x2)
        # x = self.up3(x, x1)

        # return self.out(x)
       # print("input:", x.shape)

        x1 = self.inc(x)
        # print("x1 / inc:", x1.shape)

        x2 = self.down1(x1)
        # print("x2 / down1:", x2.shape)

        x3 = self.down2(x2)
        # print("x3 / down2:", x3.shape)

        x4 = self.bottleneck(x3)
        # print("x4 / bottleneck:", x4.shape)

        x = self.up1(x4, x3)
        # print("up1:", x.shape)

        x = self.up2(x, x2)
        # print("up2:", x.shape)

        x = self.up3(x, x1)
        # print("up3:", x.shape)

        x = self.out(x)
        # print("out:", x.shape)

        return x
    
def count_params(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    model = MiniUNet(in_channels=3, num_classes=1)

    x = torch.randn(2, 3, 256, 256)
    mask = torch.randint(0, 2, (2, 1, 256, 256)).float()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

    logits = model(x)
    loss = criterion(logits, mask)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # print("logits:", logits.shape)
    # print("mask:", mask.shape)
    # print("loss:", loss.item())
    # print("number of parameters:", count_params(model))