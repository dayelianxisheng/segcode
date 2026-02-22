import torch
import torch.nn as nn
import torch.nn.functional as F


# 基础卷积块：Conv2d + BN + ReLU
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=padding, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.conv(x)


# 通道压缩模块：将不同尺度特征统一为64通道
class ChannelCompress(nn.Module):
    def __init__(self, in_channels, out_channels=64):
        super().__init__()
        self.compress = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.compress(x)


class UNet3Plus(nn.Module):
    def __init__(self, in_channels=3, num_classes=1):
        super().__init__()
        self.num_classes = num_classes

        # 编码器E1-E5
        # E1: 原图尺度 (1/1), 64通道
        self.encoder1 = ConvBlock(in_channels, 64)
        # E2: 1/2 尺度, 128通道
        self.pool1 = nn.MaxPool2d(2)
        self.encoder2 = ConvBlock(64, 128)
        # E3: 1/4 尺度, 256通道
        self.pool2 = nn.MaxPool2d(2)
        self.encoder3 = ConvBlock(128, 256)
        # E4: 1/8 尺度, 512通道
        self.pool3 = nn.MaxPool2d(2)
        self.encoder4 = ConvBlock(256, 512)
        # E5: 1/16 尺度, 1024通道（最深层）
        self.pool4 = nn.MaxPool2d(2)
        self.encoder5 = ConvBlock(512, 1024)

        # 编码器特征压缩（E1-E5 → 64通道）
        self.compress_e1 = ChannelCompress(64, 64)
        self.compress_e2 = ChannelCompress(128, 64)
        self.compress_e3 = ChannelCompress(256, 64)
        self.compress_e4 = ChannelCompress(512, 64)
        self.compress_e5 = ChannelCompress(1024, 64)

        # 解码器特征压缩（D2-D5 → 64通道，用于给浅层解码器输入）
        self.compress_d2 = ChannelCompress(320, 64)
        self.compress_d3 = ChannelCompress(320, 64)
        self.compress_d4 = ChannelCompress(320, 64)
        self.compress_d5 = ChannelCompress(320, 64)

        # 解码器D1-D5
        # D5: 1/16 尺度, 320通道 (5×64)
        self.decoder5 = ConvBlock(64 * 5, 320)
        # D4: 1/8 尺度, 320通道 (5×64)
        self.decoder4 = ConvBlock(64 * 5, 320)
        # D3: 1/4 尺度, 320通道 (5×64)
        self.decoder3 = ConvBlock(64 * 5, 320)
        # D2: 1/2 尺度, 320通道 (5×64)
        self.decoder2 = ConvBlock(64 * 5, 320)
        # D1: 原图尺度, 320通道 (5×64)
        self.decoder1 = ConvBlock(64 * 5, 320)

        # 深度监督分支（Sup）
        self.sup1 = nn.Conv2d(320, num_classes, 1)  # D1 → 输出（原图尺度）
        self.sup2 = nn.Conv2d(320, num_classes, 1)  # D2 → 上采样到原图
        self.sup3 = nn.Conv2d(320, num_classes, 1)  # D3 → 上采样到原图
        self.sup4 = nn.Conv2d(320, num_classes, 1)  # D4 → 上采样到原图
        self.sup5 = nn.Conv2d(320, num_classes, 1)  # D5 → 上采样到原图

    def forward(self, x):
        B, C, H, W = x.shape

        # 编码器前向
        # E1: 1/1 尺度, [B,64,H,W]
        e1 = self.encoder1(x)
        # E2: 1/2 尺度, [B,128,H/2,W/2]
        e2 = self.encoder2(self.pool1(e1))
        # E3: 1/4 尺度, [B,256,H/4,W/4]
        e3 = self.encoder3(self.pool2(e2))
        # E4: 1/8 尺度, [B,512,H/8,W/8]
        e4 = self.encoder4(self.pool3(e3))
        # E5: 1/16 尺度, [B,1024,H/16,W/16]
        e5 = self.encoder5(self.pool4(e4))

        # 编码器特征压缩
        e1_64 = self.compress_e1(e1)
        e2_64 = self.compress_e2(e2)
        e3_64 = self.compress_e3(e3)
        e4_64 = self.compress_e4(e4)
        e5_64 = self.compress_e5(e5)

        # 解码器D5（1/16尺度）
        # D5输入：E1(池化16倍)+E2(池化8倍)+E3(池化4倍)+E4(池化2倍)+E5(原尺度)
        e1_d5 = F.max_pool2d(e1_64, kernel_size=16, stride=16)  # [B,64,H/16,W/16]
        e2_d5 = F.max_pool2d(e2_64, kernel_size=8, stride=8)    # [B,64,H/16,W/16]
        e3_d5 = F.max_pool2d(e3_64, kernel_size=4, stride=4)    # [B,64,H/16,W/16]
        e4_d5 = F.max_pool2d(e4_64, kernel_size=2, stride=2)    # [B,64,H/16,W/16]
        e5_d5 = e5_64                                           # [B,64,H/16,W/16]
        # 拼接5个64通道特征 → 320通道
        d5_input = torch.cat([e1_d5, e2_d5, e3_d5, e4_d5, e5_d5], dim=1)
        d5 = self.decoder5(d5_input)  # [B,320,H/16,W/16]
        d5_64 = self.compress_d5(d5)  # 压缩为64通道，给D4/D3/D2/D1用

        # 解码器D4（1/8尺度）
        # D4输入：E1(池化8倍)+E2(池化4倍)+E3(池化2倍)+E4(原尺度)+D5(上采样2倍)
        e1_d4 = F.max_pool2d(e1_64, kernel_size=8, stride=8)    # [B,64,H/8,W/8]
        e2_d4 = F.max_pool2d(e2_64, kernel_size=4, stride=4)    # [B,64,H/8,W/8]
        e3_d4 = F.max_pool2d(e3_64, kernel_size=2, stride=2)    # [B,64,H/8,W/8]
        e4_d4 = e4_64                                           # [B,64,H/8,W/8]
        d5_d4 = F.interpolate(d5_64, scale_factor=2, mode='bilinear', align_corners=True)  # [B,64,H/8,W/8]
        # 拼接5个64通道特征 → 320通道
        d4_input = torch.cat([e1_d4, e2_d4, e3_d4, e4_d4, d5_d4], dim=1)
        d4 = self.decoder4(d4_input)  # [B,320,H/8,W/8]
        d4_64 = self.compress_d4(d4)  # 压缩为64通道，给D3/D2/D1用

        # 解码器D3（1/4尺度）
        # D3输入：E1(池化4倍)+E2(池化2倍)+E3(原尺度)+D4(上采样2倍)+D5(上采样4倍)
        e1_d3 = F.max_pool2d(e1_64, kernel_size=4, stride=4)    # [B,64,H/4,W/4]
        e2_d3 = F.max_pool2d(e2_64, kernel_size=2, stride=2)    # [B,64,H/4,W/4]
        e3_d3 = e3_64                                           # [B,64,H/4,W/4]
        d4_d3 = F.interpolate(d4_64, scale_factor=2, mode='bilinear', align_corners=True)  # [B,64,H/4,W/4]
        d5_d3 = F.interpolate(d5_64, scale_factor=4, mode='bilinear', align_corners=True)  # [B,64,H/4,W/4]
        # 拼接5个64通道特征 → 320通道
        d3_input = torch.cat([e1_d3, e2_d3, e3_d3, d4_d3, d5_d3], dim=1)
        d3 = self.decoder3(d3_input)  # [B,320,H/4,W/4]
        d3_64 = self.compress_d3(d3)  # 压缩为64通道，给D2/D1用

        # 解码器D2（1/2尺度）
        # D2输入：E1(池化2倍)+E2(原尺度)+D3(上采样2倍)+D4(上采样4倍)+D5(上采样8倍)
        e1_d2 = F.max_pool2d(e1_64, kernel_size=2, stride=2)    # [B,64,H/2,W/2]
        e2_d2 = e2_64                                           # [B,64,H/2,W/2]
        d3_d2 = F.interpolate(d3_64, scale_factor=2, mode='bilinear', align_corners=True)  # [B,64,H/2,W/2]
        d4_d2 = F.interpolate(d4_64, scale_factor=4, mode='bilinear', align_corners=True)  # [B,64,H/2,W/2]
        d5_d2 = F.interpolate(d5_64, scale_factor=8, mode='bilinear', align_corners=True)  # [B,64,H/2,W/2]
        # 拼接5个64通道特征 → 320通道
        d2_input = torch.cat([e1_d2, e2_d2, d3_d2, d4_d2, d5_d2], dim=1)
        d2 = self.decoder2(d2_input)  # [B,320,H/2,W/2]
        d2_64 = self.compress_d2(d2)  # 压缩为64通道，给D1用

        # 解码器D1（原图尺度）
        # D1输入：E1(原尺度)+D2(上采样2倍)+D3(上采样4倍)+D4(上采样8倍)+D5(上采样16倍)
        e1_d1 = e1_64                                           # [B,64,H,W]
        d2_d1 = F.interpolate(d2_64, scale_factor=2, mode='bilinear', align_corners=True)  # [B,64,H,W]
        d3_d1 = F.interpolate(d3_64, scale_factor=4, mode='bilinear', align_corners=True)  # [B,64,H,W]
        d4_d1 = F.interpolate(d4_64, scale_factor=8, mode='bilinear', align_corners=True)  # [B,64,H,W]
        d5_d1 = F.interpolate(d5_64, scale_factor=16, mode='bilinear', align_corners=True) # [B,64,H,W]
        # 拼接5个64通道特征 → 320通道
        d1_input = torch.cat([e1_d1, d2_d1, d3_d1, d4_d1, d5_d1], dim=1)
        d1 = self.decoder1(d1_input)  # [B,320,H,W]

        # 深度监督输出Sup
        # 所有侧输出上采样到原图尺度
        sup1 = self.sup1(d1)  # D1输出，已为原图尺度 [B,num_classes,H,W]
        sup2 = F.interpolate(self.sup2(d2), size=(H, W), mode='bilinear', align_corners=True)  # D2→原图
        sup3 = F.interpolate(self.sup3(d3), size=(H, W), mode='bilinear', align_corners=True)  # D3→原图
        sup4 = F.interpolate(self.sup4(d4), size=(H, W), mode='bilinear', align_corners=True)  # D4→原图
        sup5 = F.interpolate(self.sup5(d5), size=(H, W), mode='bilinear', align_corners=True)  # D5→原图

        # 训练时返回所有监督输出，推理时仅返回主输出sup1
        if self.training:
            return [sup1, sup2, sup3, sup4, sup5]
        else:
            return sup1

if __name__ == '__main__':
    model = UNet3Plus(in_channels=3, num_classes=1)
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        output = model(x)
    print(f"输入尺寸: {x.shape}")
    print(f"输出尺寸: {output.shape}")
