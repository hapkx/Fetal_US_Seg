"""
pvt用PVT-V2-B0预训练，没有混合encoder版

ds给的方案
"""

from abc import abstractmethod
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .fp16_util import convert_module_to_f16, convert_module_to_f32
from .nn import (
    checkpoint,
    conv_nd,
    linear,
    avg_pool_nd,
    zero_module,
    normalization,
    timestep_embedding,
)
from functools import partial
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmengine.runner import load_checkpoint
import logging




"""
pvt模块提取特征
"""
# 实现了一个简单的全连接层网络，包含两个线性层和一个激活函数（默认为 GELU）。
# 用于处理 Transformer 块中的前馈网络。
class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

# 实现了自注意力机制，包括 Q、K、V 三个线性变换
class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)     # 用于计算查询 Q 的线性层。
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)        # 用于同时计算键 K 和值 V 的线性层，输出维度是 dim * 2，因为我们同时计算 K 和 V。
        self.attn_drop = nn.Dropout(attn_drop)      # 用于同时计算键 K 和值 V 的线性层，输出维度是 dim * 2，因为我们同时计算 K 和 V。
        self.proj = nn.Linear(dim, dim)     # 将注意力的输出投影回原始维度的线性层。
        self.proj_drop = nn.Dropout(proj_drop)      # 应用在最后输出上的 Dropout 层。

        self.sr_ratio = sr_ratio        # 保存下采样比例 sr_ratio。
        # 如果下采样比例大于 1，定义一个卷积层 self.sr 用于降维，同时定义一个归一化层 self.norm 用于后续处理。
        if sr_ratio > 1:
            self.sr = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm = nn.LayerNorm(dim)

        self.window_size = 7 if sr_ratio <= 2 else 14   # 添加局部注意力窗口

    def window_partition(self, x, window_size):
        """
        将特征图划分为不重叠的局部窗口
        Args:
            x: (B, H, W, C) 输入特征图
            window_size (int): 窗口大小
        Returns:
            windows: (num_windows*B, window_size, window_size, C)
        """
        print("x.shape: ", x.shape)
        B, H, W, C = x.shape
        x = x.view(B, H//window_size, window_size, W//window_size, window_size, C)
        windows = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size, window_size, C)
        return windows
    
    def window_reverse(self, windows, window_size, H, W):
        """
        将处理后的窗口恢复为原始特征图
        Args:
            windows: (num_windows*B, window_size, window_size, C)
            window_size (int): 窗口大小
            H, W (int): 原图高宽
        Returns:
            x: (B, H, W, C)
        """
        B = int(windows.shape[0] / (H * W / window_size / window_size))
        x = windows.view(B, H//window_size, W//window_size, window_size, window_size, -1)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
        return x

    def forward(self, x, H, W):
        # # 划分窗口处理
        # x = self.window_partition(x, self.window_size)
        # # 原注意力计算
        # x = self.window_reverse(x, self.window_size, H, W)

        B, N, C = x.shape
        # 计算查询 q：通过线性层 self.q，并重塑为形状 (B, num_heads, N, head_dim)，以便后续的注意力计算。
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            # 如果 sr_ratio 大于 1，先重塑输入 x，使其形状变为 (B, C, H, W)，然后通过卷积层 self.sr 降维。
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)  # 对降维后的 x_ 进行归一化。
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)  # 计算键和值 kv，并重塑为适合注意力计算的形状。
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        # 将 kv 拆分为键 k 和值 v
        k, v = kv[0], kv[1]

        # 计算查询和键之间的点积，生成注意力分数，并进行缩放。
        # @是矩阵乘法
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)     # 对注意力分数进行 softmax 操作，得到注意力权重。
        attn = self.attn_drop(attn)     # 对注意力分数进行 softmax 操作，得到注意力权重。

        # 使用注意力权重对值 v 进行加权求和，生成输出 x，并重塑形状为 (B, N, C)。
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        # 最后，通过线性层 self.proj 投影回原始维度，并应用 Dropout。
        x = self.proj(x)
        x = self.proj_drop(x)

        return x

# 代表 Transformer 的一个基本模块，包含归一化、注意力层和 MLP 层，以及随机深度（drop path）
class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=1):
        super().__init__()
        self.norm1 = norm_layer(dim)        # 初始化第一个归一化层，使用传入的 norm_layer。
        self.attn = Attention(      # 初始化第一个归一化层，使用传入的 norm_layer。
            dim,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        # 如果 drop_path 大于 0，则使用 DropPath 对象；否则，使用 nn.Identity() 表示不进行任何操作。
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)   # 计算前馈网络的隐藏层维度 mlp_hidden_dim，是输入维度 dim 和 mlp_ratio 的乘积。
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x

# 将输入图像划分为 patch 并进行嵌入，使用卷积层将图像转换为补丁表示。
class PatchEmbed(nn.Module):
    """ Image to Patch Embedding
    """

    def __init__(self, img_size=256, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)

        self.img_size = img_size
        self.patch_size = patch_size
        assert img_size[0] % patch_size[0] == 0 and img_size[1] % patch_size[1] == 0, \
            f"img_size {img_size} should be divided by patch_size {patch_size}."
        self.H, self.W = img_size[0] // patch_size[0], img_size[1] // patch_size[1]
        self.num_patches = self.H * self.W
        # 定义一个卷积层（2D），用于将输入图像的补丁嵌入到特征空间
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        B, C, H, W = x.shape   

        # x.flatten(2): 将卷积输出的特征图展平为二维，前两个维度为批量大小和通道数，后一个维度为补丁大小。
        # x.transpose(1, 2): 转置张量，交换维度，使得补丁数（num_patches）在第二维。
        x = self.proj(x).flatten(2).transpose(1, 2)
        x = self.norm(x)
        H, W = H // self.patch_size[0], W // self.patch_size[1]

        return x, (H, W)

# 主要的 Transformer 模型。包含多个 PatchEmbedding 和 Block，支持不同阶段（stages）和不同的参数配置。具备权重初始化和位置嵌入的功能。
class PyramidVisionTransformer(nn.Module):
    def __init__(self, img_size=256, patch_size=16, in_chans=3, num_classes=3, embed_dims=[32, 64, 128, 256],
                 num_heads=[1, 2, 4, 8], mlp_ratios=[4, 4, 4, 4], qkv_bias=False, qk_scale=None, drop_rate=0.,
                 attn_drop_rate=0., drop_path_rate=0., norm_layer=nn.LayerNorm, depths=[3, 4, 6, 3],
                 sr_ratios=[8, 4, 2, 1], num_stages=4, F4=False):
        super().__init__()
        self.num_classes = num_classes
        self.depths = depths
        self.F4 = F4    # F4: 一个布尔值，控制是否返回最后的特征。
        self.num_stages = num_stages

        # 使用线性间隔生成 DropPath 的概率。dpr 是一个列表，包含每个 block 的丢弃概率。
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        cur = 0

        # 建立各个阶段
        for i in range(num_stages):
            # 为每个阶段创建补丁嵌入层 PatchEmbed。
            # 在第一个阶段，使用输入图像的大小和通道数；
            # 在其他阶段，逐渐减小图像大小和输入通道数。
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i-1)),
                                     patch_size= 1 if i == 0 else 2,
                                     in_chans=in_chans if i == 0 else embed_dims[i - 1],
                                     embed_dim=embed_dims[i])
            # 计算当前阶段的补丁数量。如果是最后一个阶段，增加一个补丁以适应特殊情况。
            num_patches = patch_embed.num_patches if i != num_stages - 1 else patch_embed.num_patches + 1
            # 创建一个可训练的参数 pos_embed，用于位置嵌入，并定义一个 dropout 层 pos_drop。
            pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dims[i]))
            pos_drop = nn.Dropout(p=drop_rate)

            # 创建一个 ModuleList，包含当前阶段的多个 Block。每个 block 的参数取决于当前阶段的配置。
            block = nn.ModuleList([Block(
                dim=embed_dims[i], num_heads=num_heads[i], mlp_ratio=mlp_ratios[i], qkv_bias=qkv_bias,
                qk_scale=qk_scale, drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[cur + j],
                norm_layer=norm_layer, sr_ratio=sr_ratios[i])
                for j in range(depths[i])])
            cur += depths[i]

            # 使用 setattr 将当前阶段的 patch_embed、pos_embed、pos_drop 和 block 以动态属性的形式添加到模型中。
            setattr(self, f"patch_embed{i + 1}", patch_embed)
            setattr(self, f"pos_embed{i + 1}", pos_embed)
            setattr(self, f"pos_drop{i + 1}", pos_drop)
            setattr(self, f"block{i + 1}", block)

            # 使用截断正态分布初始化位置嵌入的权重。
            trunc_normal_(pos_embed, std=.02)

        # 使用PVTv2-B0的官方配置参数
        self.depths = [2, 2, 2, 2]
        self.embed_dims = [32, 64, 160, 256]
        self.num_heads = [1, 2, 5, 8]
        self.mlp_ratios = [8, 8, 4, 4]
        self.sr_ratios = [8, 4, 2, 1]
        
        # init weights
        self.apply(self._init_weights)

    # 初始化权重方法
    # def init_weights(self, pretrained=None):
    #     if isinstance(pretrained, str):
    #         # logger = get_root_logger()
    #         logger = logging.getLogger()
    #         load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)
    def _init_weights(self, m):
        # 保持与预训练模型一致的初始化方式
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

    # 定义一个私有方法 _init_weights，用于初始化模型中的不同层的权重。
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    # 获取位置嵌入
    def _get_pos_embed(self, pos_embed, patch_embed, H, W):
        # 如果当前的高和宽与第一个 patch 嵌入的补丁数量匹配，则直接返回位置嵌入。
        if H * W == self.patch_embed1.num_patches:
            return pos_embed
        else:
            # 否则，使用双线性插值调整位置嵌入的大小以匹配当前阶段的补丁。
            return F.interpolate(
                pos_embed.reshape(1, patch_embed.H, patch_embed.W, -1).permute(0, 3, 1, 2),
                size=(H, W), mode="bilinear").reshape(1, -1, H * W).permute(0, 2, 1)

    # 前向特征提取
    def forward_features(self, x):
        # 初始化一个空列表 outs 用于存储每个阶段的输出，B 是当前批量大小。
        outs = []
        B = x.shape[0]

        # 遍历每个阶段，逐步处理输入特征。
        for i in range(self.num_stages):
            patch_embed = getattr(self, f"patch_embed{i + 1}")
            pos_embed = getattr(self, f"pos_embed{i + 1}")
            pos_drop = getattr(self, f"pos_drop{i + 1}")
            block = getattr(self, f"block{i + 1}")
            # 将输入 x 通过补丁嵌入层，得到嵌入的特征以及高和宽。
            x, (H, W) = patch_embed(x)
            # 获取合适的位置信息。如果是最后一个阶段，略过第一个位置嵌入。
            if i == self.num_stages - 1:
                pos_embed = self._get_pos_embed(pos_embed[:, 1:], patch_embed, H, W)
            else:
                pos_embed = self._get_pos_embed(pos_embed, patch_embed, H, W)
            # 将位置嵌入添加到嵌入特征上并应用 dropout。
            x = pos_drop(x + pos_embed)
            # 通过当前阶段的所有 Transformer block 处理特征。
            for blk in block:
                x = blk(x, H, W)
            # 将特征重塑为适合 CNN 处理的格式，并调整维度顺序。
            x = x.reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
            outs.append(x)

        return outs

    def forward(self, x):
        x = self.forward_features(x)
        if self.F4:
            x = x[3:4]
        return x
    

# 这是一个带有注意力机制的池化层。它使用位置编码和 QKV (Query, Key, Value) 机制进行自注意力计算，从而在空间维度上进行聚合特征。输入张量的形状被调整以适应自注意力的计算。
class AttentionPool2d(nn.Module):
    """
    Adapted from CLIP: https://github.com/openai/CLIP/blob/main/clip/model.py
    """

    def __init__(
        self,
        spacial_dim: int,
        embed_dim: int,
        num_heads_channels: int,
        output_dim: int = None,
    ):
        super().__init__()
        self.positional_embedding = nn.Parameter(
            torch.randn(embed_dim, spacial_dim ** 2 + 1) / embed_dim ** 0.5
        )
        self.qkv_proj = conv_nd(1, embed_dim, 3 * embed_dim, 1)
        self.c_proj = conv_nd(1, embed_dim, output_dim or embed_dim, 1)
        self.num_heads = embed_dim // num_heads_channels
        self.attention = QKVAttention(self.num_heads)

    def forward(self, x):
        b, c, *_spatial = x.shape
        
        x = x.reshape(b, c, -1)  # NC(HW)
        x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(HW+1)
        x = x + self.positional_embedding[None, :, :].to(x.dtype)  # NC(HW+1)
        x = self.qkv_proj(x)
        x = self.attention(x)
        x = self.c_proj(x)
        return x[:, :, 0]

# 抽象基类，任何需要时间步嵌入作为第二个输入的模块都可以继承此类。
class TimestepBlock(nn.Module):
    """
    Any module where forward() takes timestep embeddings as a second argument.
    """

    @abstractmethod
    def forward(self, x, emb):
        """
        Apply the module to `x` given `emb` timestep embeddings.
        """

# 继承自 nn.Sequential，可以将时间步嵌入（embeddings）传递给所有支持此功能的子模块。
class TimestepEmbedSequential(nn.Sequential, TimestepBlock):
    """
    A sequential module that passes timestep embeddings to the children that
    support it as an extra input.
    """

    def forward(self, x, emb):
        for layer in self:
            if isinstance(layer, TimestepBlock):
                x = layer(x, emb)
            else:
                x = layer(x)
        return x

class Upsample(nn.Module):
    """
    An upsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 upsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        if use_conv:
            self.conv = conv_nd(dims, self.channels, self.out_channels, 3, padding=1)

    def forward(self, x):
        assert x.shape[1] == self.channels
        if self.dims == 3:
            x = F.interpolate(  # F.interpolate 用于最近邻插值上采样。
                x, (x.shape[2], x.shape[3] * 2, x.shape[4] * 2), mode="nearest"
            )
        else:
            x = F.interpolate(x, scale_factor=2, mode="nearest")
        if self.use_conv:
            x = self.conv(x)
        return x

class Downsample(nn.Module):
    """
    A downsampling layer with an optional convolution.

    :param channels: channels in the inputs and outputs.
    :param use_conv: a bool determining if a convolution is applied.
    :param dims: determines if the signal is 1D, 2D, or 3D. If 3D, then
                 downsampling occurs in the inner-two dimensions.
    """

    def __init__(self, channels, use_conv, dims=2, out_channels=None):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.dims = dims
        stride = 2 if dims != 3 else (1, 2, 2)
        if use_conv:
            self.op = conv_nd(
                dims, self.channels, self.out_channels, 3, stride=stride, padding=1
            )
        else:
            assert self.channels == self.out_channels
            self.op = avg_pool_nd(dims, kernel_size=stride, stride=stride)

    def forward(self, x):
        assert x.shape[1] == self.channels
        return self.op(x)

class ResBlock(TimestepBlock):
    """
    A residual block that can optionally change the number of channels.

    :param channels: the number of input channels.
    :param emb_channels: the number of timestep embedding channels.
    :param dropout: the rate of dropout.
    :param out_channels: if specified, the number of out channels.
    :param use_conv: if True and out_channels is specified, use a spatial
        convolution instead of a smaller 1x1 convolution to change the
        channels in the skip connection.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param use_checkpoint: if True, use gradient checkpointing on this module.
    :param up: if True, use this block for upsampling.
    :param down: if True, use this block for downsampling.
    """

    def __init__(
        self,
        channels,
        emb_channels,
        dropout,
        out_channels=None,
        use_conv=False,
        use_scale_shift_norm=False,
        dims=2,
        use_checkpoint=False,
        up=False,
        down=False,
    ):
        super().__init__()
        self.channels = channels
        self.emb_channels = emb_channels
        self.dropout = dropout
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_checkpoint = use_checkpoint
        self.use_scale_shift_norm = use_scale_shift_norm
        # 定义输入层序列，包括归一化、激活函数（SiLU）和卷积层。
        self.in_layers = nn.Sequential(
            normalization(channels),
            nn.SiLU(),
            conv_nd(dims, channels, self.out_channels, 3, padding=1),
        )

        # 标记是否进行上采样或下采样。
        self.updown = up or down

        # 创建上采样或下采样实例，
        # 如果 up 为真，使用 Upsample 类进行上采样；
        # 如果 down 为真，使用 Downsample 类进行下采样；
        # 否则使用 nn.Identity() 表示不做任何处理。创建上采样或下采样实例，如果 up 为真，使用 Upsample 类进行上采样；如果 down 为真，使用 Downsample 类进行下采样；否则使用 nn.Identity() 表示不做任何处理。
        if up:
            self.h_upd = Upsample(channels, False, dims)
            self.x_upd = Upsample(channels, False, dims)
        elif down:
            self.h_upd = Downsample(channels, False, dims)
            self.x_upd = Downsample(channels, False, dims)
        else:
            self.h_upd = self.x_upd = nn.Identity()
            # nn.Identity() 代表一个恒等映射（identity mapping），在前向传播过程中不对输入做任何处理，直接返回输入。

        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                2 * self.out_channels if use_scale_shift_norm else self.out_channels,
            ),
        )
        self.out_layers = nn.Sequential(
            normalization(self.out_channels),
            nn.SiLU(),
            nn.Dropout(p=dropout),
            zero_module(
                conv_nd(dims, self.out_channels, self.out_channels, 3, padding=1)
            ),
        )

        if self.out_channels == channels:
            self.skip_connection = nn.Identity()
        elif use_conv:
            self.skip_connection = conv_nd(
                dims, channels, self.out_channels, 3, padding=1
            )
        else:
            self.skip_connection = conv_nd(dims, channels, self.out_channels, 1)

    def forward(self, x, emb):
        """
        Apply the block to a Tensor, conditioned on a timestep embedding.

        :param x: an [N x C x ...] Tensor of features.
        :param emb: an [N x emb_channels] Tensor of timestep embeddings.
        :return: an [N x C x ...] Tensor of outputs.
        """
        return checkpoint(
            self._forward, (x, emb), self.parameters(), self.use_checkpoint
        )

    
    def _forward(self, x, emb):
        if self.updown:
            in_rest, in_conv = self.in_layers[:-1], self.in_layers[-1]
            h = in_rest(x)
            h = self.h_upd(h)
            x = self.x_upd(x)
            h = in_conv(h)
        else:
            h = self.in_layers(x)
        emb_out = self.emb_layers(emb).type(h.dtype)
        while len(emb_out.shape) < len(h.shape):
            emb_out = emb_out[..., None]
        if self.use_scale_shift_norm:
            out_norm, out_rest = self.out_layers[0], self.out_layers[1:]
            scale, shift = torch.chunk(emb_out, 2, dim=1)
            h = out_norm(h) * (1 + scale) + shift
            h = out_rest(h)
        else:
            h = h + emb_out
            h = self.out_layers(h)
        return self.skip_connection(x) + h

class AttentionBlock(nn.Module):
    """
    An attention block that allows spatial positions to attend to each other.

    Originally ported from here, but adapted to the N-d case.
    https://github.com/hojonathanho/diffusion/blob/1e0dceb3b3495bbe19116a5e1b3596cd0706c543/diffusion_tf/models/unet.py#L66.
    """

    def __init__(
        self,
        channels,
        num_heads=1,
        num_head_channels=-1,
        use_checkpoint=False,
        use_new_attention_order=False,
    ):
        super().__init__()
        self.channels = channels
        if num_head_channels == -1:
            self.num_heads = num_heads
        else:
            assert (
                channels % num_head_channels == 0
            ), f"q,k,v channels {channels} is not divisible by num_head_channels {num_head_channels}"
            self.num_heads = channels // num_head_channels
        self.use_checkpoint = use_checkpoint
        self.norm = normalization(channels)
        self.qkv = conv_nd(1, channels, channels * 3, 1)
        if use_new_attention_order:
            # split qkv before split heads
            self.attention = QKVAttention(self.num_heads)
        else:
            # split heads before split qkv
            self.attention = QKVAttentionLegacy(self.num_heads)

        self.proj_out = zero_module(conv_nd(1, channels, channels, 1))

    def forward(self, x):
        return checkpoint(self._forward, (x,), self.parameters(), True)

    def _forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1)
        qkv = self.qkv(self.norm(x))
        h = self.attention(qkv)
        h = self.proj_out(h)
        return (x + h).reshape(b, c, *spatial)


"""
encoder模块
"""

class EncoderBlock(TimestepBlock):
    """
    A UNet encoder block consisting of a double-layer residual (DLR) component
    followed by an adaptive feature selection (AFS) part.

    :param in_channels: the number of input channels.
    :param out_channels: the number of output channels.
    :param reduction_ratio: the reduction ratio for the channel attention module.
    """

    def __init__(self, in_channels, out_channels, emb_channels, reduction_ratio=16):
        super(EncoderBlock, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.reduction_ratio = reduction_ratio

        # Double-layer residual (DLR) component
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.shortcut_conv = nn.Conv2d(out_channels, out_channels, kernel_size=1) 

        # Adaptive feature selection (AFS) component
        self.channel_attention = ChannelAttention(out_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention()

        # Time embedding layers
        self.emb_layers = nn.Sequential(
            nn.SiLU(),
            linear(
                emb_channels,
                self.out_channels,
            ),
        )

    def forward(self, x, emb):
        x1 = F.relu(self.bn1(self.conv1(x)))
        # print("x1.shape: ", x1.shape)
        x2 = x1 * self.channel_attention(x1)
        x3 = self.shortcut_conv(x1) + x2
        x4 = x3 * self.spatial_attention(x3)
        x5 = self.shortcut_conv(x3) + x4
        x5 = F.relu(self.bn2(self.conv2(x5)))

        emb_out = self.emb_layers(emb).type(x5.dtype)
        while len(emb_out.shape) < len(x5.shape):
            emb_out = emb_out[..., None]
        x5 = x5 + emb_out  # Embedding is added to the feature map

        # Final output with residual connection
        return x5 

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio, bias=False),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        batch, channels, _, _ = x.size()
        avg_out = self.mlp(self.avg_pool(x).view(batch, channels))
        max_out = self.mlp(self.max_pool(x).view(batch, channels))
        out = avg_out + max_out
        return self.sigmoid(out).view(batch, channels, 1, 1)

class SpatialAttention(nn.Module):
    def __init__(self):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv(x)
        return self.sigmoid(x)


"""
unet 整体结构
"""

class MutualCrossAttention(nn.Module):
    def __init__(self, channels, num_heads=1, dropout=0.0):
        super(MutualCrossAttention, self).__init__()
        self.channels = channels
        self.num_heads = num_heads
        self.dropout = nn.Dropout(dropout)

        # Define layers for query, key, and value
        self.qkv = nn.Linear(channels, channels * 3)  # Similar to conv_nd but for linear
        self.attention = nn.MultiheadAttention(channels, num_heads, dropout=dropout)
        self.norm = nn.LayerNorm(channels)
        self.proj_out = nn.Linear(channels, channels)

    def forward(self, x):
        b, c, *spatial = x.shape
        x = x.reshape(b, c, -1).permute(2, 0, 1)  # Rearrange for nn.MultiheadAttention
        qkv = self.qkv(self.norm(x))
        q, k, v = torch.chunk(qkv, 3, dim=-1)
        
        # Use PyTorch's built-in MultiheadAttention
        attn_output, _ = self.attention(q, k, v)
        attn_output = self.proj_out(attn_output)

        return (x + attn_output).permute(1, 2, 0).reshape(b, c, *spatial)

def count_flops_attn(model, _x, y):
    """
    A counter for the `thop` package to count the operations in an
    attention operation.
    Meant to be used like:
        macs, params = thop.profile(
            model,
            inputs=(inputs, timestamps),
            custom_ops={QKVAttention: QKVAttention.count_flops},
        )
    """
    b, c, *spatial = y[0].shape
    num_spatial = int(np.prod(spatial))
    # We perform two matmuls with the same number of ops.
    # The first computes the weight matrix, the second computes
    # the combination of the value vectors.
    matmul_ops = 2 * b * (num_spatial ** 2) * c
    model.total_ops += torch.DoubleTensor([matmul_ops])

class QKVAttentionLegacy(nn.Module):
    """
    A module which performs QKV attention. Matches legacy QKVAttention + input/ouput heads shaping
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (H * 3 * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.reshape(bs * self.n_heads, ch * 3, length).split(ch, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts", q * scale, k * scale
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v)
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class QKVAttention(nn.Module):
    """
    A module which performs QKV attention and splits in a different order.
    """

    def __init__(self, n_heads):
        super().__init__()
        self.n_heads = n_heads

    def forward(self, qkv):
        """
        Apply QKV attention.

        :param qkv: an [N x (3 * H * C) x T] tensor of Qs, Ks, and Vs.
        :return: an [N x (H * C) x T] tensor after attention.
        """
        bs, width, length = qkv.shape
        assert width % (3 * self.n_heads) == 0
        ch = width // (3 * self.n_heads)
        q, k, v = qkv.chunk(3, dim=1)
        scale = 1 / math.sqrt(math.sqrt(ch))
        weight = torch.einsum(
            "bct,bcs->bts",
            (q * scale).view(bs * self.n_heads, ch, length),
            (k * scale).view(bs * self.n_heads, ch, length),
        )  # More stable with f16 than dividing afterwards
        weight = torch.softmax(weight.float(), dim=-1).type(weight.dtype)
        a = torch.einsum("bts,bcs->bct", weight, v.reshape(bs * self.n_heads, ch, length))
        return a.reshape(bs, -1, length)

    @staticmethod
    def count_flops(model, _x, y):
        return count_flops_attn(model, _x, y)

class UNetModel(nn.Module):
    """
    The full UNet model with attention and timestep embedding.

    :param in_channels: channels in the input Tensor.
    :param model_channels: base channel count for the model.
    :param out_channels: channels in the output Tensor.
    :param num_res_blocks: number of residual blocks per downsample.
    :param attention_resolutions: a collection of downsample rates at which
        attention will take place. May be a set, list, or tuple.
        For example, if this contains 4, then at 4x downsampling, attention
        will be used.
    :param dropout: the dropout probability.
    :param channel_mult: channel multiplier for each level of the UNet.
    :param conv_resample: if True, use learned convolutions for upsampling and
        downsampling.
    :param dims: determines if the signal is 1D, 2D, or 3D.
    :param num_classes: if specified (as an int), then this model will be
        class-conditional with `num_classes` classes.
    :param use_checkpoint: use gradient checkpointing to reduce memory usage.
    :param num_heads: the number of attention heads in each attention layer.
    :param num_heads_channels: if specified, ignore num_heads and instead use
                               a fixed channel width per attention head.
    :param num_heads_upsample: works with num_heads to set a different number
                               of heads for upsampling. Deprecated.
    :param use_scale_shift_norm: use a FiLM-like conditioning mechanism.
    :param resblock_updown: use residual blocks for up/downsampling.
    :param use_new_attention_order: use a different attention pattern for potentially
                                    increased efficiency.
    image_size: 输入图像的大小。
    in_channels: 输入特征的通道数。
    model_channels: 模型的基础通道数。
    out_channels: 输出特征的通道数。
    num_res_blocks: 每个下采样阶段的残差块数量。
    attention_resolutions: 注意力模块应用的分辨率。
    dropout: dropout 概率。
    channel_mult: 不同层级的通道倍增因子。
    conv_resample: 是否使用卷积进行上下采样。
    dims: 数据的维度（1D, 2D, 3D）。
    num_classes: 类别数（用于类别条件模型）。
    use_checkpoint: 是否使用梯度检查点。
    num_heads: 每个注意力层的头数。
    num_head_channels: 每个头的固定通道宽度。
    num_heads_upsample: 用于上采样的注意力头数。
    use_scale_shift_norm: 是否使用 FiLM 类似的条件机制。
    resblock_updown: 是否使用残差块进行上下采样。
    use_new_attention_order: 是否使用新的注意力模式。

    """

    def __init__(
        self,
        image_size,
        in_channels,
        model_channels,
        num_res_blocks,
        attention_resolutions,
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
        dims=2,
        num_classes=3,
        use_checkpoint=False,
        use_fp16=False,
        num_heads=1,
        num_head_channels=-1,
        num_heads_upsample=-1,
        use_scale_shift_norm=False,
        resblock_updown=False,
        use_new_attention_order=False,
        pvt = None,
        out_channels=3,
        high_way = True,
    ):
        super().__init__()

        if num_heads_upsample == -1:
            num_heads_upsample = num_heads

        self.image_size = image_size
        self.in_channels = in_channels
        self.model_channels = model_channels
        self.out_channels = num_classes
        self.num_res_blocks = num_res_blocks
        self.attention_resolutions = attention_resolutions
        self.dropout = dropout
        self.channel_mult = channel_mult
        self.conv_resample = conv_resample
        self.num_classes = num_classes
        self.use_checkpoint = use_checkpoint
        self.dtype = torch.float16 if use_fp16 else torch.float32
        self.num_heads = num_heads
        self.num_head_channels = num_head_channels
        self.num_heads_upsample = num_heads_upsample

        # 时间步嵌入
        time_embed_dim = model_channels * 4
        self.time_embed = nn.Sequential(
            linear(model_channels, time_embed_dim),
            nn.SiLU(),  # 激活函数
            linear(time_embed_dim, time_embed_dim),
        )

        self.pvt = pvt
        # 加载预训练权重
        self.load_pvt_weights(pretrained='pvt_v2_b0.pth')

        # 如果模型是类别条件的，定义一个嵌入层，将类别转换为对应的嵌入。
        if self.num_classes is not None:
            self.label_emb = nn.Embedding(num_classes, time_embed_dim)

        # 输入块。第一层是卷积层，使用 TimestepEmbedSequential 包裹，以便传递时间步嵌入。后续的输入块会根据网络的深度不断增加。
        self.input_blocks = nn.ModuleList(
            [
                TimestepEmbedSequential(
                    conv_nd(dims, in_channels, model_channels, 3, padding=1)
                )
            ]
        )
        # 初始化特征大小，用于记录当前特征图的通道数。
        self._feature_size = model_channels
        input_block_chans = [model_channels]
        ch = model_channels
        ds = 1
        # 循环构建每个分辨率级别的网络层级，通道数通过 channel_mult 控制。
        # channel_mult 是一个元组，定义了在不同层级的特征通道数的倍增因子。
        # 通过这个循环，你可以控制每个分辨率级别的通道数。
        # 例如，如果 channel_mult 是 (1, 2, 4, 8)，那么模型的通道数将在不同层级上依次为 model_channels * 1, model_channels * 2, model_channels * 4, model_channels * 8。
        for level, mult in enumerate(channel_mult):
            for _ in range(num_res_blocks):
                layers = [
                    # EncoderBlock(
                    #     ch,
                    #     mult * model_channels,
                    #     time_embed_dim
                    # )
                    ResBlock(
                        ch,
                        time_embed_dim,
                        dropout,
                        out_channels=mult * model_channels,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = mult * model_channels
                # 如果当前分辨率在指定的注意力分辨率中，则插入一个注意力块。
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                self.input_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            # 如果不是最后一级的分辨率，插入一个下采样模块，可能是残差块或普通下采样。
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(
                    TimestepEmbedSequential(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            down=True,
                        )
                        if resblock_updown
                        else Downsample(
                            ch, conv_resample, dims=dims, out_channels=out_ch
                        )
                    )
                )
                ch = out_ch
                input_block_chans.append(ch)
                ds *= 2
                self._feature_size += ch

        # 处理特征的中间块，包含残差块和注意力块，增强模型的特征提取能力。
        self.middle_block = TimestepEmbedSequential(
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
            AttentionBlock(
                ch,
                use_checkpoint=use_checkpoint,
                num_heads=num_heads,
                num_head_channels=num_head_channels,
                use_new_attention_order=use_new_attention_order,
            ),
            ResBlock(
                ch,
                time_embed_dim,
                dropout,
                dims=dims,
                use_checkpoint=use_checkpoint,
                use_scale_shift_norm=use_scale_shift_norm,
            ),
        )
        self._feature_size += ch

        # 中间块用于连接编码器和解码器部分，包含残差块和注意力块。
        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(num_res_blocks + 1):
                ich = input_block_chans.pop()
                layers = [
                    ResBlock(
                        ch + ich,
                        time_embed_dim,
                        dropout,
                        out_channels=model_channels * mult,
                        dims=dims,
                        use_checkpoint=use_checkpoint,
                        use_scale_shift_norm=use_scale_shift_norm,
                    )
                ]
                ch = model_channels * mult
                if ds in attention_resolutions:
                    layers.append(
                        AttentionBlock(
                            ch,
                            use_checkpoint=use_checkpoint,
                            num_heads=num_heads_upsample,
                            num_head_channels=num_head_channels,
                            use_new_attention_order=use_new_attention_order,
                        )
                    )
                if level and i == num_res_blocks:
                    out_ch = ch
                    layers.append(
                        ResBlock(
                            ch,
                            time_embed_dim,
                            dropout,
                            out_channels=out_ch,
                            dims=dims,
                            use_checkpoint=use_checkpoint,
                            use_scale_shift_norm=use_scale_shift_norm,
                            up=True,
                        )
                        if resblock_updown
                        else Upsample(ch, conv_resample, dims=dims, out_channels=out_ch)
                    )
                    ds //= 2
                self.output_blocks.append(TimestepEmbedSequential(*layers))
                self._feature_size += ch

        # 定义输出层，将特征转换为最终输出，包括归一化、激活和一个卷积层。
        self.out = nn.Sequential(
            normalization(ch),
            nn.SiLU(),
            zero_module(conv_nd(dims, model_channels, out_channels, 3, padding=1)),
            # nn.Softmax(dim=1) if self.num_classes > 1 else nn.Sigmoid()  # 多分类用Softmax
        )
        # self.align_conv = nn.Conv2d(x.size(1), pvt_feat.size(1), 1)
        # 添加PVT特征处理模块
        self.pvt_adjust = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(128, 128, 1),
                nn.GroupNorm(32, 128)
            ),
            nn.Sequential(
                nn.Conv2d(256, 128, 1),
                nn.GroupNorm(32, 128)
            ),
            nn.Sequential(
                nn.Conv2d(512, 256, 1),
                nn.GroupNorm(32, 256)
            ),
            nn.Sequential(
                nn.Conv2d(1024, 512, 1),
                nn.GroupNorm(32, 512)
            )
        ])
        
        # 可学习的融合权重
        self.fusion_weights = nn.Parameter(torch.ones(4))
        

    def load_pvt_weights(self, pretrained):
        if pretrained:
            state_dict = torch.load(pretrained)['model']
            
            # 转换键名映射
            new_state_dict = {}
            for k, v in state_dict.items():
                if k.startswith('block'):
                    # 将block转换为stage结构
                    stage_num = int(k.split('.')[0][-1]) - 1
                    new_k = k.replace(f'block{stage_num+1}', f'block{stage_num}')
                elif k.startswith('patch_embed'):
                    stage_num = int(k.split('patch_embed')[1][0])
                    new_k = k.replace(f'patch_embed{stage_num}', f'patch_embed{stage_num+1}')
                else:
                    new_k = k
                new_state_dict[new_k] = v
            
            # 严格加载权重
            self.pvt.load_state_dict(new_state_dict, strict=False)
            print(f"Loaded pretrained weights from {pretrained}")

    
    # 供了转换模型数据类型的方法，以支持不同的精度计算（FP16 或 FP32）。
    def convert_to_fp16(self):
        """
        Convert the torso of the model to float16.
        """
        self.input_blocks.apply(convert_module_to_f16)
        self.middle_block.apply(convert_module_to_f16)
        self.output_blocks.apply(convert_module_to_f16)

    def convert_to_fp32(self):
        """
        Convert the torso of the model to float32.
        """
        self.input_blocks.apply(convert_module_to_f32)
        self.middle_block.apply(convert_module_to_f32)
        self.output_blocks.apply(convert_module_to_f32)

    def highway_forward(self,x, hs):
        return self.hwm(x,hs)
    
    def _align_features(self, pvt_feat, target_size, index):
        """
        pvt_feat: PVT输出的特征图
        target_size: 目标尺寸(H,W)
        index: PVT层级索引(0-3)
        """
        # 通道对齐
        feat = self.pvt_adjust[index](pvt_feat)
        # 空间尺寸对齐
        if feat.shape[-2:] != target_size:
            feat = F.interpolate(feat, size=target_size, mode='bilinear', align_corners=False)
        return feat
    
    def forward(self, x, timesteps, y=None):
        """
        Apply the model to an input batch.

        :param x: an [N x C x ...] Tensor of inputs.
        :param timesteps: a 1-D batch of timesteps.
        :param y: an [N] Tensor of labels, if class-conditional.
        :return: an [N x C x ...] Tensor of outputs.
        """
        if y is not None:
            emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
            emb = emb + self.label_emb(y)  # 融合标签信息
            
        hs = []     # 每一处特征图
        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels))
        # 记录各阶段特征尺寸
        feature_sizes = []

        h = x.type(self.dtype)  # h.shape: torch.Size([4, 4, 256, 256])
        c = h[:,:-3,...]    # c.shape: torch.Size([4, 3, 256, 256])

        # 定义各融合阶段的索引映射
        fusion_stages = [3, 6, 9, 12]  # 对应4个下采样后的位置
        
        pvt_features = self.pvt(c)     
        # pvt_f[0]: torch.Size([1, 128, 256, 256])
        # pvt_f[1]: torch.Size([1, 256, 128, 128])
        # pvt_f[2]: torch.Size([1, 512, 64, 64])
        # pvt_f[3]: torch.Size([1, 1024, 32, 32])

        for ind, module in enumerate(self.input_blocks):                
            h = module(h, emb)
            # 在下采样层后执行融合
            if ind in fusion_stages:
                # 获取当前层级索引(0-3)
                pvt_level = fusion_stages.index(ind)
                # 获取当前特征尺寸
                current_size = h.shape[2:]
                # 调整PVT特征
                adjusted_feat = self._align_features(
                    pvt_features[pvt_level], 
                    current_size,
                    pvt_level
                )
                # 加权融合
                fusion_weight = torch.sigmoid(self.fusion_weights[pvt_level])
                h = h * (1 - fusion_weight) + adjusted_feat * fusion_weight
                hs.append(h)
                
                # 记录特征尺寸
                feature_sizes.append(current_size)
            
            else:
                hs.append(h)

        h = self.middle_block(h, emb)
        for module in self.output_blocks:
            h = torch.cat([h, hs.pop()], dim=1)
            h = module(h, emb)
        h = h.type(x.dtype)
        out = self.out(h)
        return out
    
    def _fuse(self, x, pvt_feat):
        # # 添加通道对齐卷积
        # align_conv = nn.Conv2d(x.size(1), pvt_feat.size(1), 1).to(x.device)
        # # add * 
        # return align_conv(x) * torch.sigmoid(pvt_feat)  # 门控融合
        return x * pvt_feat


