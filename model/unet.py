"""
UNet模型定义
"""
import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    """卷积块: Conv -> BN -> Dropout -> LeakyReLU"""
    def __init__(self, in_channels, out_channels):
        super(ConvBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.3),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class DownSampleBlock(nn.Module):
    """下采样块"""
    def __init__(self, channel):
        super(DownSampleBlock, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(channel, channel, 3, 2, 1, padding_mode='reflect', bias=False),
            nn.BatchNorm2d(channel),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.layer(x)


class UpSampleBlock(nn.Module):
    """上采样块"""
    def __init__(self, channel):
        super(UpSampleBlock, self).__init__()
        self.layer = nn.Conv2d(channel, channel // 2, 1, 1)

    def forward(self, x, feature_map):
        up = F.interpolate(x, scale_factor=2, mode='nearest')
        out = self.layer(up)
        return torch.cat([feature_map, out], 1)


class UNet(nn.Module):
    """UNet模型"""
    def __init__(self, in_channels=3, num_classes=1):
        super(UNet, self).__init__()

        # 编码器
        self.c1 = ConvBlock(in_channels, 64)
        self.d1 = DownSampleBlock(64)
        self.c2 = ConvBlock(64, 128)
        self.d2 = DownSampleBlock(128)
        self.c3 = ConvBlock(128, 256)
        self.d3 = DownSampleBlock(256)
        self.c4 = ConvBlock(256, 512)
        self.d4 = DownSampleBlock(512)
        self.c5 = ConvBlock(512, 1024)

        # 解码器
        self.u1 = UpSampleBlock(1024)
        self.c6 = ConvBlock(1024, 512)
        self.u2 = UpSampleBlock(512)
        self.c7 = ConvBlock(512, 256)
        self.u3 = UpSampleBlock(256)
        self.c8 = ConvBlock(256, 128)
        self.u4 = UpSampleBlock(128)
        self.c9 = ConvBlock(128, 64)

        # 输出
        self.out = nn.Conv2d(64, num_classes, 3, 1, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 编码
        r1 = self.c1(x)
        r2 = self.c2(self.d1(r1))
        r3 = self.c3(self.d2(r2))
        r4 = self.c4(self.d3(r3))
        r5 = self.c5(self.d4(r4))

        # 解码
        o1 = self.c6(self.u1(r5, r4))
        o2 = self.c7(self.u2(o1, r3))
        o3 = self.c8(self.u3(o2, r2))
        o4 = self.c9(self.u4(o3, r1))

        return self.sigmoid(self.out(o4))


if __name__ == '__main__':
    x = torch.randn(2, 3, 256, 256)
    net = UNet()
    print(net(x).shape)
