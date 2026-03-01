import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from torchvision.models import resnet50, ResNet50_Weights


ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu}


class Conv2dReLU(nn.Sequential):
    """卷积-BN-ReLU块"""
    def __init__(self, in_channels, out_channels, kernel_size, padding=0, stride=1, use_batchnorm=True):
        conv = nn.Conv2d(in_channels, out_channels, kernel_size, stride=stride, padding=padding, bias=not use_batchnorm)
        relu = nn.ReLU(inplace=True)
        bn = nn.BatchNorm2d(out_channels) if use_batchnorm else nn.Identity()
        super(Conv2dReLU, self).__init__(conv, bn, relu)


class ResNetSkipConnections(nn.Module):
    """
    ResNet骨干网络 - 用于提取跳跃连接特征

    从ResNet的不同层提取特征，用于解码器的跳跃连接
    """
    def __init__(self, pretrained=True):
        super(ResNetSkipConnections, self).__init__()

        # 加载预训练ResNet50
        if pretrained:
            resnet = resnet50(weights=ResNet50_Weights.IMAGENET1K_V1)
        else:
            resnet = resnet50(weights=None)

        # ResNet的层
        self.conv1 = resnet.conv1      # 1/2, 64
        self.bn1 = resnet.bn1
        self.relu = resnet.relu
        self.maxpool = resnet.maxpool  # 1/4, 64

        self.layer1 = resnet.layer1    # 1/4, 256
        self.layer2 = resnet.layer2    # 1/8, 512
        self.layer3 = resnet.layer3    # 1/16, 1024
        self.layer4 = resnet.layer4    # 1/32, 2048

    def forward(self, x):
        """
        前向传播，返回多尺度特征

        Returns:
            features: 特征列表 [skip1, skip2, skip3, final]
                skip1: 1/4分辨率, 64通道
                skip2: 1/8分辨率, 256通道
                skip3: 1/16分辨率, 512通道
                final: 1/32分辨率, 1024通道
        """
        # 初始卷积
        x = self.conv1(x)    # (B, 64, H/2, W/2)
        x = self.bn1(x)
        x = self.relu(x)
        x0 = x  # 1/2特征（可选使用）

        x = self.maxpool(x)  # (B, 64, H/4, W/4)

        # 各层特征
        x1 = self.layer1(x)  # (B, 256, H/4, W/4)  ← Skip 1
        x2 = self.layer2(x1) # (B, 512, H/8, W/8)   ← Skip 2
        x3 = self.layer3(x2) # (B, 1024, H/16, W/16) ← Skip 3

        # 对于TransUNet，我们使用1/16分辨率作为Transformer输入
        # 这里返回用于跳跃连接的特征
        return [x1, x2, x3], x3


class PatchEmbedding(nn.Module):
    """
    图像块嵌入层

    将特征图分割为patches并嵌入到向量空间
    """
    def __init__(self, in_channels=1024, embed_dim=768, patch_size=1):
        super(PatchEmbedding, self).__init__()
        # patch_size=1 表示不进行额外的patch分割，直接投影
        self.projection = Conv2d(
            in_channels=in_channels,
            out_channels=embed_dim,
            kernel_size=patch_size,
            stride=patch_size
        )

    def forward(self, x):
        """
        Args:
            x: 特征图 (B, C, H, W)

        Returns:
            patches: (B, N, embed_dim) 其中 N = H × W
        """
        x = self.projection(x)  # (B, embed_dim, H, W)
        x = x.flatten(2)        # (B, embed_dim, H*W)
        x = x.transpose(-1, -2) # (B, H*W, embed_dim)
        return x


class MultiHeadAttention(nn.Module):
    """多头自注意力机制"""
    def __init__(self, embed_dim=768, num_heads=12, attention_dropout_rate=0.0):
        super(MultiHeadAttention, self).__init__()
        self.num_attention_heads = num_heads
        self.attention_head_size = int(embed_dim / num_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(embed_dim, self.all_head_size)
        self.key = Linear(embed_dim, self.all_head_size)
        self.value = Linear(embed_dim, self.all_head_size)

        self.out = Linear(embed_dim, embed_dim)
        self.attn_dropout = Dropout(attention_dropout_rate)
        self.proj_dropout = Dropout(attention_dropout_rate)

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output


class MLP(nn.Module):
    """Transformer的MLP前馈网络"""
    def __init__(self, embed_dim=768, mlp_dim=3072, dropout_rate=0.1):
        super(MLP, self).__init__()
        self.fc1 = Linear(embed_dim, mlp_dim)
        self.fc2 = Linear(mlp_dim, embed_dim)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(dropout_rate)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer编码器块 - Attention + MLP"""
    def __init__(self, embed_dim=768, mlp_dim=3072, num_heads=12, dropout_rate=0.1, attention_dropout_rate=0.0):
        super(TransformerBlock, self).__init__()
        self.hidden_size = embed_dim
        self.attention_norm = LayerNorm(embed_dim, eps=1e-6)
        self.ffn_norm = LayerNorm(embed_dim, eps=1e-6)
        self.ffn = MLP(embed_dim, mlp_dim, dropout_rate)
        self.attn = MultiHeadAttention(embed_dim, num_heads, attention_dropout_rate)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x


class TransformerEncoder(nn.Module):
    """Transformer编码器 - 堆叠多个Transformer块"""
    def __init__(self, embed_dim=768, mlp_dim=3072, num_heads=12, num_layers=12,
                 dropout_rate=0.1, attention_dropout_rate=0.0):
        super(TransformerEncoder, self).__init__()
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(embed_dim, eps=1e-6)
        for _ in range(num_layers):
            layer = TransformerBlock(embed_dim, mlp_dim, num_heads, dropout_rate, attention_dropout_rate)
            self.layer.append(layer)

    def forward(self, hidden_states):
        for layer_block in self.layer:
            hidden_states = layer_block(hidden_states)
        encoded = self.encoder_norm(hidden_states)
        return encoded


class DecoderBlock(nn.Module):
    """
    解码器块 - 上采样 + 跳跃连接融合

    架构:
        上采样(×2) → 拼接跳跃连接 → Conv3×3 → Conv3×3
    """
    def __init__(self, in_channels, out_channels, skip_channels=0, use_batchnorm=True):
        super().__init__()
        # 上采样后拼接
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        """
        Args:
            x: 上层特征 (B, C_in, H, W)
            skip: 跳跃连接特征 (B, C_skip, 2H, 2W)

        Returns:
            out: 融合后的特征 (B, C_out, 2H, 2W)
        """
        x = self.up(x)
        if skip is not None:
            # 确保尺寸匹配
            if x.size()[2:] != skip.size()[2:]:
                skip = F.interpolate(skip, size=x.size()[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup(nn.Module):
    """
    UNet风格的解码器

    将Transformer编码器的输出逐步上采样，并与ResNet特征融合

    Args:
        embed_dim: Transformer嵌入维度
        decoder_channels: 各解码器层的通道数 (例如: (256, 128, 64, 16))
        skip_channels: 各跳跃连接的通道数 (例如: (256, 512, 1024))
    """
    def __init__(self, embed_dim=768, decoder_channels=(256, 128, 64, 16), skip_channels=(256, 512, 1024)):
        super().__init__()
        # 将Transformer输出转换为解码器输入
        head_channels = 512
        self.conv_more = Conv2dReLU(
            embed_dim,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True
        )

        # 构建解码器块
        # in_channels: head_channels + decoder_channels[:-1]
        # 例如: [512, 256, 128, 64]
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        # 确保 skip_channels 长度匹配，不足的补0
        # 例如: (256, 512, 1024) → (256, 512, 1024, 0)
        num_blocks = len(in_channels)
        if len(skip_channels) < num_blocks:
            skip_channels = list(skip_channels) + [0] * (num_blocks - len(skip_channels))

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch)
            for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, skip_features):
        """
        Args:
            hidden_states: Transformer编码器输出 (B, n_patches, embed_dim)
            skip_features: ResNet跳跃连接特征 [skip1, skip2, skip3]

        Returns:
            x: 解码后的特征图 (B, decoder_channels[-1], H, W)
        """
        B, n_patch, hidden = hidden_states.size()
        h, w = int(math.sqrt(n_patch)), int(math.sqrt(n_patch))

        # 重塑为2D特征图
        x = hidden_states.permute(0, 2, 1)        # (B, embed_dim, n_patches)
        x = x.contiguous().view(B, hidden, h, w)   # (B, embed_dim, h, w)
        x = self.conv_more(x)

        # 逐层上采样并融合跳跃连接
        for i, decoder_block in enumerate(self.blocks):
            skip = skip_features[i] if i < len(skip_features) else None
            x = decoder_block(x, skip=skip)

        return x


class SegmentationHead(nn.Sequential):
    """
    分割头

    最终的1×1卷积层，将特征映射到类别数
    """
    def __init__(self, in_channels, out_channels, kernel_size=3):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        super().__init__(conv2d)


class TransUNet(nn.Module):
    """
    TransUNet模型 - 用于图像分割任务

    架构: ResNet(跳跃连接) → PatchEmbedding → Transformer → Decoder → SegHead

    Args:
        img_size: 输入图像尺寸
        in_channels: 输入图像通道数
        num_classes: 输出类别数
        embed_dim: Transformer嵌入维度
        mlp_dim: MLP隐藏层维度
        num_heads: 多头注意力的头数
        num_layers: Transformer层数
        decoder_channels: 解码器各层通道数
        dropout_rate: Dropout率
        attention_dropout_rate: 注意力Dropout率
        pretrained_resnet: 是否使用预训练ResNet
    """
    def __init__(self, img_size=224, in_channels=3, num_classes=1,
                 embed_dim=768, mlp_dim=3072, num_heads=12, num_layers=12,
                 decoder_channels=(256, 128, 64, 16),
                 dropout_rate=0.1, attention_dropout_rate=0.0, pretrained_resnet=True):
        super(TransUNet, self).__init__()
        self.num_classes = num_classes
        self.img_size = _pair(img_size)

        # 1. ResNet骨干网络 - 提取跳跃连接特征
        self.resnet = ResNetSkipConnections(pretrained=pretrained_resnet)

        # 2. Patch Embedding - 将ResNet输出转换为patch序列
        # ResNet layer3输出是1024通道，作为Transformer输入
        self.patch_embedding = PatchEmbedding(
            in_channels=1024,
            embed_dim=embed_dim,
            patch_size=1
        )

        # 3. Position Embedding
        # 对于224×224输入，layer3输出是14×14（H/16, W/16）
        self.position_embeddings = nn.Parameter(torch.zeros(1, 14 * 14, embed_dim))
        self.dropout = Dropout(dropout_rate)

        # 4. Transformer Encoder
        self.transformer_encoder = TransformerEncoder(
            embed_dim=embed_dim,
            mlp_dim=mlp_dim,
            num_heads=num_heads,
            num_layers=num_layers,
            dropout_rate=dropout_rate,
            attention_dropout_rate=attention_dropout_rate
        )

        # 5. Decoder
        # skip_channels: ResNet layer1(256), layer2(512), layer3(1024)
        self.decoder = DecoderCup(
            embed_dim=embed_dim,
            decoder_channels=decoder_channels,
            skip_channels=(256, 512, 1024)
        )

        # 6. Segmentation Head
        self.segmentation_head = SegmentationHead(
            in_channels=decoder_channels[-1],
            out_channels=num_classes,
            kernel_size=3
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """
        前向传播

        Args:
            x: 输入图像 (B, C, H, W)

        Returns:
            output: 分割预测 (B, num_classes, H, W)
        """
        input_size = x.size()[2:]

        # 处理单通道输入 - 复制为3通道
        if x.size()[1] == 1:
            x = x.repeat(1, 3, 1, 1)

        # 1. ResNet提取跳跃连接特征
        skip_features, resnet_output = self.resnet(x)

        # 2. Patch Embedding
        x = self.patch_embedding(resnet_output)

        # 3. Add Position Embedding
        # 处理不同输入尺寸
        B, N, C = x.shape
        if N != self.position_embeddings.size(1):
            # 插值位置编码
            pos_embed = self.position_embeddings.transpose(1, 2).reshape(1, C, 14, 14)
            h = w = int(math.sqrt(N))
            pos_embed = F.interpolate(pos_embed, size=(h, w), mode='bilinear', align_corners=False)
            pos_embed = pos_embed.reshape(1, C, h * w).transpose(1, 2)
            x = x + pos_embed
        else:
            x = x + self.position_embeddings

        x = self.dropout(x)

        # 4. Transformer Encoding
        encoded = self.transformer_encoder(x)

        # 5. Decoding with Skip Connections
        x = self.decoder(encoded, skip_features)

        # 6. Segmentation
        logits = self.segmentation_head(x)

        # 7. 上采样到原始尺寸
        if logits.size()[2:] != input_size:
            logits = F.interpolate(logits, size=input_size, mode='bilinear', align_corners=False)

        return self.sigmoid(logits)


class TransUNetConfig:
    """TransUNet配置类 - 预定义配置"""
    @staticmethod
    def vit_b16_config():
        """ViT-B/16配置"""
        return {
            'embed_dim': 768,
            'mlp_dim': 3072,
            'num_heads': 12,
            'num_layers': 12,
            'decoder_channels': (256, 128, 64, 16),
            'dropout_rate': 0.1,
            'attention_dropout_rate': 0.0
        }

    @staticmethod
    def vit_l16_config():
        """ViT-L/16配置"""
        return {
            'embed_dim': 1024,
            'mlp_dim': 4096,
            'num_heads': 16,
            'num_layers': 24,
            'decoder_channels': (256, 128, 64, 16),
            'dropout_rate': 0.1,
            'attention_dropout_rate': 0.0
        }


def create_transunet(variant='vit_b16', img_size=224, in_channels=3, num_classes=1, **kwargs):
    """
    创建TransUNet模型的便捷函数

    Args:
        variant: 模型变体 ('vit_b16', 'vit_l16')
        img_size: 输入图像尺寸
        in_channels: 输入通道数
        num_classes: 输出类别数
        **kwargs: 其他参数覆盖

    Returns:
        TransUNet模型
    """
    config_dict = {
        'vit_b16': TransUNetConfig.vit_b16_config(),
        'vit_l16': TransUNetConfig.vit_l16_config()
    }

    config = config_dict.get(variant, TransUNetConfig.vit_b16_config())
    config.update(kwargs)

    return TransUNet(
        img_size=img_size,
        in_channels=in_channels,
        num_classes=num_classes,
        **config
    )


if __name__ == '__main__':
    # 测试TransUNet
    print("=" * 60)
    print("测试TransUNet模型")
    print("=" * 60)

    # 创建模型
    model = create_transunet(variant='vit_b16', img_size=224, in_channels=3, num_classes=1)

    # 计算参数量
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"模型参数量: {total_params:,}")
    print(f"可训练参数量: {trainable_params:,}")

    # 测试前向传播
    x = torch.randn(2, 3, 224, 224)
    output = model(x)
    print(f"输入形状: {x.shape}")
    print(f"输出形状: {output.shape}")

    # 测试单通道输入
    x_gray = torch.randn(1, 1, 224, 224)
    output_gray = model(x_gray)
    print(f"单通道输入形状: {x_gray.shape}")
    print(f"单通道输出形状: {output_gray.shape}")

    print("\nTransUNet模型测试通过!")
