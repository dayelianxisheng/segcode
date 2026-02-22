
"""DeepLabV3 for binary segmentation"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet50, resnet101
from torchvision.models.resnet import ResNet50_Weights, ResNet101_Weights


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, dilation=1):
        super().__init__()
        self.cbr = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.cbr(x)


class ASPP(nn.Module):
    """Atrous Spatial Pyramid Pooling"""
    def __init__(self, in_channels, out_channels=256):
        super().__init__()
        self.branch1 = ConvBnReLU(in_channels, out_channels, 1, 1, 0)
        self.branch2 = ConvBnReLU(in_channels, out_channels, 3, 1, 6, 6)
        self.branch3 = ConvBnReLU(in_channels, out_channels, 3, 1, 12, 12)
        self.branch4 = ConvBnReLU(in_channels, out_channels, 3, 1, 18, 18)
        self.branch5 = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            ConvBnReLU(in_channels, out_channels, 1, 1, 0)
        )

    def forward(self, x):
        size = x.size()[2:]
        return torch.cat([
            self.branch1(x),
            self.branch2(x),
            self.branch3(x),
            self.branch4(x),
            F.interpolate(self.branch5(x), size, mode="nearest")
        ], dim=1)


class DeepLabV3(nn.Module):
    def __init__(self, num_classes=1, backbone="resnet50", pretrained=True):
        super().__init__()
        self.backbone_name = backbone
        
        if backbone == "resnet50":
            weights = ResNet50_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = resnet50(weights=weights)
        elif backbone == "resnet101":
            weights = ResNet101_Weights.IMAGENET1K_V1 if pretrained else None
            resnet = resnet101(weights=weights)
        else:
            raise ValueError("backbone must be resnet50 or resnet101")

        # Extract ResNet layers
        self.conv1 = nn.Sequential(resnet.conv1, resnet.bn1, resnet.relu)
        self.maxpool = resnet.maxpool
        self.layer1 = resnet.layer1  # 256 channels, 1/4
        self.layer2 = resnet.layer2  # 512 channels, 1/8
        self.layer3 = resnet.layer3  # 1024 channels, 1/16
        self.layer4 = resnet.layer4  # 2048 channels, 1/32

        # Low level features for skip connection
        self.low_conv = ConvBnReLU(256, 48, 1, 1, 0)

        # ASPP module on high-level features
        self.aspp = ASPP(2048, 256)
        self.aspp_conv = nn.Sequential(
            ConvBnReLU(256 * 5, 256, 1, 1, 0),
            nn.Dropout2d(0.5)
        )

        # Final fusion and classification
        self.cat_conv = nn.Sequential(
            ConvBnReLU(48 + 256, 256, 3, 1, 1),
            nn.Conv2d(256, num_classes, 1)
        )
        
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        input_size = x.size()[2:]
        
        # Encoder
        x = self.conv1(x)
        x = self.maxpool(x)           # 1/4
        low = self.layer1(x)           # 256, 1/4
        x = self.layer2(low)           # 512, 1/8
        x = self.layer3(x)            # 1024, 1/16
        high = self.layer4(x)          # 2048, 1/32
        
        # ASPP on high-level features
        aspp = self.aspp(high)
        aspp = self.aspp_conv(aspp)
        
        # Upsample ASPP to low feature resolution (8 -> 64, factor=8)
        aspp_up = F.interpolate(aspp, size=low.size()[2:], mode="bilinear", align_corners=True)
        
        # Process low-level features
        low = self.low_conv(low)
        
        # Concatenate and classify
        cat = torch.cat([aspp_up, low], dim=1)
        cat = self.cat_conv(cat)
        
        # Upsample to input resolution
        out = F.interpolate(cat, size=input_size, mode="bilinear", align_corners=True)
        
        return self.sigmoid(out)


if __name__ == "__main__":
    import torch
    net = DeepLabV3(num_classes=1, backbone="resnet50", pretrained=False)
    x = torch.rand(2, 3, 256, 256)
    y = net(x)
    print(f"Input: {x.shape}, Output: {y.shape}")
