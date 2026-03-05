import math
import warnings
import numpy as np
from functools import partial

import torch
import torch.nn as nn
from sympy import num_digits


def _no_grad_trunc_normal_(tensor, mean, std, a, b):
    # pytorch源码 正态分布初始化
    def norm_cdf(x):
        """
        计算标准正态分布的积累分布函数
        """
        return (1. + math.erf(x / math.sqrt(2.))) / 2.

    if (mean < a - 2 * std) or (mean > b + 2 * std):
        warnings.warn("均值距离 [a, b] 范围超过了 2 个标准差。"
                      "nn.init.trunc_normal_ 生成的数值分布可能会不准确。",
                      stacklevel=2)

    with torch.no_grad():
        l = norm_cdf((a - mean) / std)
        u = norm_cdf((b - mean) / std)

        tensor.uniform_(2 * l - 1, 2 * u - 1)

        tensor.erfinv_()

        tensor.mul_(std * math.sqrt(2.))
        tensor.add_(mean)

        tensor.clamp_(min=a, max=b)
        return tensor


def trunc_normal_(tensor, mean=0., std=1., a=-2., b=2.):
    """
    主要用于在不计算梯度的情况下初始化张量
    """
    return _no_grad_trunc_normal_(tensor, mean, std, a, b)


# 激活函数
class GELU(nn.Module):
    def __init__(self):
        super(GELU, self).__init__()

    def forward(self, x):
        return 0.5 * x * (1 + torch.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * torch.pow(x, 3))))


class OverlabPatchEmbed(nn.Module):
    def __init__(self, patch_size=7, stride=4, in_chans=3, embed_dim=768):
        super(OverlabPatchEmbed, self).__init__()
        patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride)
        self.norm = nn.LayerNorm(embed_dim)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        # 卷积投影：[B, C, H, W] -> [B, embed_dim, H / stride, W / stride]
        x = self.proj(x)
        _, _, h, w = x.shape
        # 展平并转置：[B, C', H', W'] -> [B, H'*W', C']
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x, h, w


class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super(Attention, self).__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.sr_ratio = sr_ratio
        if self.sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        # x=(B, 16384, 32)
        B,N,C=x.shape

        # 生成 Query (q): # (B, 16384, 32) -> (B, 16384, 8, 4) -> (B, 8, 16384, 4)
        q = self.q(x).reshape(B,N,self.num_headsC//self.num_heads).permute(0,2,1,3)

        if self.sr_ratio > 1:
            # 1. 还原 2D 图像: (B, 16384, 32) -> (B, 32, 128, 128)
            x_ = x.permute(0,2,1).reshape(B,C,H,W)

            # 2. 核心压缩: (B, 32, 128, 128) -> (B, 32, 16, 16) 使用 stride=8 的卷积，像素减少了 8x8=64 倍
            x_ = self.sr(x_)

            # 3. 序列化: (B, 32, 16, 16) -> (B, 32, 256) -> (B, 256, 32)
            x_ = x_.reshape(B,C,-1).permute(0,2,1)

            # 4. 归一化
            x_ = self.norm(x_)

            # 生成 KV: (B, 256, 32) -> (B, 256, 64) -> (2, B, 8, 256, 4)
            # 输出维度翻倍（32 变 64），因为包含 k 和 v
            # 拆分成 2 个分量（k/v）、8 个头、每个头 4 维
            # 把索引 2（即代表 k 或 v 的维度）提到最前面
            kv=self.kv(x_).reshape(B,-1,2,self.num_heads,C//self.num_heads).permute(0,2,3,1,4)
        else:
            kv = self.kv(x).reshape(B,-1,2,self.num_heads,C// self.num_heads).permute(0,2,3,1,4)

        # k和 v 的形状均为 (B, 8, 256, 4)
        k,v = kv[0],kv[1]

        # q: [B, 8, 16384, 4] @ k.T: [B, 8, 4, 256] -> attn: [B, 8, 16384, 256]
        # 矩阵乘法 (16384, 4) @ (4, 256) = (16384, 256)
        # 每个高清像素（16384个）都在全局缩略信息（256个）中寻找相关性
        attn = (q @ k.transpose(-2,-1)) * self.scale

        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        # attn: [B, 8, 16384, 256] @ v: [B, 8, 256, 4] -> x: [B, 8, 16384, 4]
        x = (attn @ v).transpose(-1,-2).reshape(B,N,C)

        # 投影
        x = self.proj(x)
        x = self.proj_drop(x)
        return x

