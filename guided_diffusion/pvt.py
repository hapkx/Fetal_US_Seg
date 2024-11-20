import torch
import torch.nn as nn
import torch.nn.functional as F
from functools import partial

from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg
from mmseg.models.builder import BACKBONES
from mmseg.utils import get_root_logger
from mmcv.runner import load_checkpoint


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

    def forward(self, x, H, W):
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

    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
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
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dims=[64, 128, 256, 512],
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
            patch_embed = PatchEmbed(img_size=img_size if i == 0 else img_size // (2 ** (i + 1)),
                                     patch_size=patch_size if i == 0 else 2,
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

        # init weights
        self.apply(self._init_weights)

    # 初始化权重方法
    def init_weights(self, pretrained=None):
        if isinstance(pretrained, str):
            logger = get_root_logger()
            load_checkpoint(self, pretrained, map_location='cpu', strict=False, logger=logger)

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


def _conv_filter(state_dict, patch_size=16):
    """ convert patch embedding weight from manual patchify + linear proj to conv"""
    out_dict = {}
    for k, v in state_dict.items():
        if 'patch_embed.proj.weight' in k:
            v = v.reshape((v.shape[0], 3, patch_size, patch_size))
        out_dict[k] = v

    return out_dict


@BACKBONES.register_module()
class pvt_tiny(PyramidVisionTransformer):
    def __init__(self, **kwargs):
        super(pvt_tiny, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2],
            sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)


@BACKBONES.register_module()
class pvt_small(PyramidVisionTransformer):
    def __init__(self, **kwargs):
        super(pvt_small, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3],
            sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)


@BACKBONES.register_module()
class pvt_medium(PyramidVisionTransformer):
    def __init__(self, **kwargs):
        super(pvt_medium, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 18, 3],
            sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)


@BACKBONES.register_module()
class pvt_large(PyramidVisionTransformer):
    def __init__(self, **kwargs):
        super(pvt_large, self).__init__(
            patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4],
            qkv_bias=True, norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 8, 27, 3],
            sr_ratios=[8, 4, 2, 1], drop_rate=0.0, drop_path_rate=0.1)


