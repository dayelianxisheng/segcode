
import torch
import torch.nn as nn
# 论文：MogaNet: Multi-order Gated Aggregation Network (ICLR 2024)
# 论文地址：https://arxiv.org/pdf/2211.03295
# Github地址：https://github.com/Westlake-AI/MogaNet
# 全网最全100➕即插即用模块GitHub地址：https://github.com/ai-dawang/PlugNPlay-Modules
# FFN with Channel Aggregation

def build_act_layer(act_type):
    #Build activation layer
    if act_type is None:
        return nn.Identity()
    assert act_type in ['GELU', 'ReLU', 'SiLU']
    if act_type == 'SiLU':
        return nn.SiLU()
    elif act_type == 'ReLU':
        return nn.ReLU()
    else:
        return nn.GELU()

class ElementScale(nn.Module):
    #A learnable element-wise scaler.

    def __init__(self, embed_dims, init_value=0., requires_grad=True):
        super(ElementScale, self).__init__()
        self.scale = nn.Parameter(
            init_value * torch.ones((1, embed_dims, 1, 1)),
            requires_grad=requires_grad
        )

    def forward(self, x):
        return x * self.scale

class ChannelAggregationFFN(nn.Module):
    """带有通道聚合的前馈网络
        embed_dims (int): 特征维度。通常与 MultiheadAttention（多头注意力机制）的维度一致。
        feedforward_channels (int): FFN 的隐藏层维度。
        kernel_size (int): 深度卷积（Depth-wise Conv）的卷积核大小。默认值为 3。
        act_type (str): 激活函数的类型。默认值为 'GELU'。
        ffn_drop (float, 可选): FFN 中元素被置零（Dropout）的概率。默认值为 0.0。
    """

    def __init__(self,
                 embed_dims,
                 kernel_size=3,
                 act_type='GELU',
                 ffn_drop=0.):
        super(ChannelAggregationFFN, self).__init__()

        self.embed_dims = embed_dims
        self.feedforward_channels = int(embed_dims * 4)

        self.fc1 = nn.Conv2d(
            in_channels=embed_dims,
            out_channels=self.feedforward_channels,
            kernel_size=1)
        self.dwconv = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=self.feedforward_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            bias=True,
            groups=self.feedforward_channels)
        self.act = build_act_layer(act_type)
        self.fc2 = nn.Conv2d(
            in_channels=self.feedforward_channels,
            out_channels=embed_dims,
            kernel_size=1)
        self.drop = nn.Dropout(ffn_drop)

        self.decompose = nn.Conv2d(
            in_channels=self.feedforward_channels,  # C -> 1
            out_channels=1, kernel_size=1,
        )
        self.sigma = ElementScale(
            self.feedforward_channels, init_value=1e-5, requires_grad=True)
        self.decompose_act = build_act_layer(act_type)

    def feat_decompose(self, x):
        # x_d: [B, C, H, W] -> [B, 1, H, W]
        x = x + self.sigma(x - self.decompose_act(self.decompose(x)))
        return x

    def forward(self, x, H, W):
        B,N,C=x.shape
        x = x.transpose(1,2).view(B,C,H,W).contiguous()
        # proj 1
        x = self.fc1(x)
        x = self.dwconv(x)
        x = self.act(x)
        x = self.drop(x)
        # proj 2
        x = self.feat_decompose(x)
        x = self.fc2(x)
        x = self.drop(x)
        x = x.flatten(2).transpose(1, 2).contiguous()
        return x


if __name__ == '__main__':
    B,C,H,W = 1,64,32, 32
    N = H * W

    # 1. 构造输入: [B, N, C] -> [1, 1024, 64]
    dummy_input = torch.randn(B, N, C)
    block = ChannelAggregationFFN(embed_dims=C)

    # 3. 运行前向传播，传入 H, W
    output = block(dummy_input, H, W)

    # 4. 打印结果
    print(f"输入形状 (B, N, C): {dummy_input.size()}")
    print(f"输出形状 (B, N, C): {output.size()}")

