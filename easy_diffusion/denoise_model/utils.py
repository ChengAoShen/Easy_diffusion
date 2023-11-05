import torch
from torch import nn
import math
from inspect import isfunction
from einops.layers.torch import Rearrange


def default(val, d):
    if val is not None:
        return val
    return d() if isfunction(d) else d


class Residual(nn.Module):
    def __init__(self, fn):
        """
        残差连接模块
        :param fn: 激活函数类型
        """
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        """
        残差连接前馈
        :param x: 输入数据
        :param args:
        :param kwargs:
        :return: f(x) + x
        """
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    """
    这个上采样模块的作用是将输入张量的尺寸在宽和高上放大 2 倍
    :param dim:
    :param dim_out:
    :return:
    """
    return nn.Sequential(
        nn.Upsample(scale_factor=2, mode="nearest"),            # 先使用最近邻填充将数据在长宽上翻倍
        nn.Conv2d(dim, default(dim_out, dim), 3, padding=1),    # 再使用卷积对翻倍后的数据提取局部相关关系填充
    )


def Downsample(dim, dim_out=None):
    # No More Strided Convolutions or Pooling
    """
    下采样模块的作用是将输入张量的分辨率降低，通常用于在深度学习模型中对特征图进行降采样。
    在这个实现中，下采样操作的方式是使用一个 $2 \times 2$ 的最大池化操作，
    将输入张量的宽和高都缩小一半，然后再使用上述的变换和卷积操作得到输出张量。
    由于这个实现使用了形状变换操作，因此没有使用传统的卷积或池化操作进行下采样，
    从而避免了在下采样过程中丢失信息的问题。
    :param dim:
    :param dim_out:
    :return:
    """
    return nn.Sequential(
        # 将输入张量的形状由 (batch_size, channel, height, width) 变换为 (batch_size, channel * 4, height / 2, width / 2)
        Rearrange("b c (h p1) (w p2) -> b (c p1 p2) h w", p1=2, p2=2),
        # 对变换后的张量进行一个 $1 \times 1$ 的卷积操作，将通道数从 dim * 4（即变换后的通道数）降到 dim（即指定的输出通道数），得到输出张量。
        nn.Conv2d(dim * 4, default(dim_out, dim), 1),
    )


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings