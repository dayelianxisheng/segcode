import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101
from torchvision.models.resnet import ResNet50_Weights, ResNet101_Weights


class ConvBnReLU(nn.Module):
    """卷积-BN-ReLU块"""
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.cbr(x)


class PositionAttentionModule(nn.Module):
    """位置注意力模块 (Position Attention Module)
    捕获空间位置之间的依赖关系
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        # Query, Key, Value 卷积
        self.query_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.key_conv = nn.Conv2d(in_channels, in_channels // 8, kernel_size=1)
        self.value_conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)

        # 可学习权重参数
        self.gamma = nn.Parameter(torch.zeros(1))

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            out: 注意力加权后的特征 + 原始特征
        """
        batch_size, channels, height, width = x.size()

        # Query: (B, C//8, H, W) -> (B, H*W, C//8)
        proj_query = self.query_conv(x).view(batch_size, -1, width * height).permute(0, 2, 1)

        # Key: (B, C//8, H, W) -> (B, C//8, H*W)
        proj_key = self.key_conv(x).view(batch_size, -1, width * height)

        # Energy: (B, H*W, H*W) - 空间注意力图
        energy = torch.bmm(proj_query, proj_key)
        attention = self.softmax(energy)

        # Value: (B, C, H, W) -> (B, C, H*W)
        proj_value = self.value_conv(x).view(batch_size, -1, width * height)

        # Out: (B, C, H*W) -> (B, C, H, W)
        out = torch.bmm(proj_value, attention.permute(0, 2, 1))
        out = out.view(batch_size, channels, height, width)

        # 残差连接
        out = self.gamma * out + x
        return out


class ChannelAttentionModule(nn.Module):
    """通道注意力模块 (Channel Attention Module)
    捕获通道之间的依赖关系
    """
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels
        self.gamma = nn.Parameter(torch.zeros(1))
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x):
        """
        Args:
            x: 输入特征图 (B, C, H, W)
        Returns:
            out: 注意力加权后的特征 + 原始特征
        """
        batch_size, channels, height, width = x.size()

        # Query: (B, C, H*W)
        proj_query = x.view(batch_size, channels, -1)

        # Key: (B, H*W, C)
        proj_key = x.view(batch_size, channels, -1).permute(0, 2, 1)

        # Energy: (B, C, C) - 通道注意力图
        energy = torch.bmm(proj_query, proj_key)

        # 使用 max-energy 增强区分度
        energy_new = torch.max(energy, -1, keepdim=True)[0].expand_as(energy) - energy
        attention = self.softmax(energy_new)

        # Value: (B, C, H*W)
        proj_value = x.view(batch_size, channels, -1)

        # Out: (B, C, H*W) -> (B, C, H, W)
        out = torch.bmm(attention, proj_value)
        out = out.view(batch_size, channels, height, width)

        # 残差连接
        out = self.gamma * out + x
        return out


class DANetHead(nn.Module):
    """DANet分割头
    包含位置注意力分支、通道注意力分支和融合分支
    """
    def __init__(self, in_channels, num_classes, norm_layer=nn.BatchNorm2d):
        super().__init__()
        inter_channels = in_channels // 4

        # 位置注意力分支
        self.pa_conv = ConvBnReLU(in_channels, inter_channels, kernel_size=3, padding=1)
        self.pa_module = PositionAttentionModule(inter_channels)
        self.pa_conv_out = ConvBnReLU(inter_channels, inter_channels, kernel_size=3, padding=1)
        self.pa_cls = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(inter_channels, num_classes, kernel_size=1)
        )

        # 通道注意力分支
        self.ca_conv = ConvBnReLU(in_channels, inter_channels, kernel_size=3, padding=1)
        self.ca_module = ChannelAttentionModule(inter_channels)
        self.ca_conv_out = ConvBnReLU(inter_channels, inter_channels, kernel_size=3, padding=1)
        self.ca_cls = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(inter_channels, num_classes, kernel_size=1)
        )

        # 融合分支
        self.fusion_cls = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(inter_channels, num_classes, kernel_size=1)
        )

    def forward(self, x):
        """
        Args:
            x: 输入特征图
        Returns:
            fusion_out: 融合分支输出 (主输出)
            pa_out: 位置注意力分支输出
            ca_out: 通道注意力分支输出
        """
        # 位置注意力分支
        pa_feat = self.pa_conv(x)
        pa_feat = self.pa_module(pa_feat)
        pa_feat = self.pa_conv_out(pa_feat)
        pa_out = self.pa_cls(pa_feat)

        # 通道注意力分支
        ca_feat = self.ca_conv(x)
        ca_feat = self.ca_module(ca_feat)
        ca_feat = self.ca_conv_out(ca_feat)
        ca_out = self.ca_cls(ca_feat)

        # 融合分支
        fusion_feat = pa_feat + ca_feat
        fusion_out = self.fusion_cls(fusion_feat)

        return fusion_out, pa_out, ca_out


class DANet(nn.Module):
    """Dual Attention Network for Binary Segmentation
    结合位置注意力和通道注意力的语义分割网络
    """
    def __init__(self, num_classes=1, backbone='resnet50', pretrained=True):
        super().__init__()
        self.num_classes = num_classes
        self.backbone_name = backbone

        # Backbone
        if backbone == 'resnet50':
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = resnet50(weights=weights)
            backbone_channels = 2048
        elif backbone == 'resnet101':
            weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = resnet101(weights=weights)
            backbone_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        # 提取ResNet层
        self.layer0 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu, resnet.maxpool)
        self.layer1 = resnet.layer1  # 256 channels
        self.layer2 = resnet.layer2  # 512 channels
        self.layer3 = resnet.layer3  # 1024 channels
        self.layer4 = resnet.layer4  # 2048 channels

        # DANet Head
        self.head = DANetHead(backbone_channels, num_classes)

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """前向传播
        Args:
            x: 输入图像 (B, 3, H, W)
        Returns:
            fusion_out: 融合分支的分割结果
            pa_out: 位置注意力分支的分割结果
            ca_out: 通道注意力分支的分割结果
        """
        input_size = x.size()[2:]

        # Encoder
        x = self.layer0(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        # DANet Head
        fusion_out, pa_out, ca_out = self.head(x)

        # 上采样到输入尺寸
        fusion_out = F.interpolate(fusion_out, size=input_size, mode='bilinear', align_corners=True)
        pa_out = F.interpolate(pa_out, size=input_size, mode='bilinear', align_corners=True)
        ca_out = F.interpolate(ca_out, size=input_size, mode='bilinear', align_corners=True)

        # Sigmoid激活
        fusion_out = self.sigmoid(fusion_out)
        pa_out = self.sigmoid(pa_out)
        ca_out = self.sigmoid(ca_out)

        return fusion_out, pa_out, ca_out


class DANetSimple(nn.Module):
    """DANet简化版 - 只输出融合分支结果
    适用于只需要主输出的场景
    """
    def __init__(self, num_classes=1, backbone='resnet50', pretrained=True):
        super().__init__()
        self.danet = DANet(num_classes, backbone, pretrained)

    def forward(self, x):
        fusion_out, _, _ = self.danet(x)
        return fusion_out


if __name__ == "__main__":
    import torch

    # 测试 DANet
    print("Testing DANet...")
    model = DANet(num_classes=1, backbone='resnet50', pretrained=False)
    x = torch.randn(2, 3, 256, 256)
    fusion_out, pa_out, ca_out = model(x)
    print(f"Input: {x.shape}")
    print(f"Fusion Output: {fusion_out.shape}")
    print(f"PA Output: {pa_out.shape}")
    print(f"CA Output: {ca_out.shape}")

    # 测试 DANetSimple
    print("\nTesting DANetSimple...")
    model_simple = DANetSimple(num_classes=1, backbone='resnet50', pretrained=False)
    out = model_simple(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
