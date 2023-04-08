# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import time


class SegHead(nn.Module):

    def __init__(self, in_chans, img_size=(512, 832)):
        super().__init__()

        self.img_size = img_size
        self.conv = nn.Conv2d(in_chans, 2, kernel_size=1)
        self.conv1 = nn.Conv2d(2, 2, 1)

    def forward(self, x):
        x = self.conv(x)
        _, _, h, w = x.shape
        x = F.interpolate(x, size=(h * 2, w * 4), mode='bilinear', align_corners=True)
        x = self.conv1(x)
        x = F.interpolate(x, size=(h * 4, w * 8), mode='bilinear', align_corners=True)
        x = self.conv1(x)
        x = F.interpolate(x, size=(h * 8, w * 16), mode='bilinear', align_corners=True)
        x = self.conv1(x)
        x = F.interpolate(x, size=self.img_size, mode='bilinear', align_corners=True)
        return x


class Mlp(nn.Module):
    def __init__(
            self,
            in_features,
            hidden_features=None,
            out_features=None,
            act_layer=nn.GELU,
            drop=0.0,
    ):
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


class Attention(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            qk_scale=None,
            attn_drop=0.0,
            proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
                .reshape(B, N, 3, self.num_heads, C // self.num_heads)
                .permute(2, 0, 3, 1, 4)
        )
        q, k, v = (
            qkv[0],
            qkv[1],
            qkv[2],
        )  # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            qk_scale=qk_scale,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x):
        _x = self.attn(self.norm1(x))
        x = x + self.drop_path(_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention2(nn.Module):
    def __init__(
            self,
            dim,
            num_heads=8,
            qkv_bias=False,
            attn_drop=0.0,
            proj_drop=0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = head_dim ** -0.5

        self.to_q = nn.Linear(dim, dim, bias=False)
        self.to_k = nn.Linear(dim, dim, bias=True)
        self.to_v = nn.Linear(dim, dim, bias=True)

        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, target):
        B, N, C = target.shape
        q = rearrange(self.to_q(target), 'b n (h d) -> b h n d', h=self.num_heads)
        k = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h=self.num_heads)
        v = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h=self.num_heads)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class Block2(nn.Module):
    def __init__(
            self,
            dim,
            num_heads,
            mlp_ratio=4.0,
            qkv_bias=False,
            qk_scale=None,
            drop=0.0,
            attn_drop=0.0,
            drop_path=0.0,
            act_layer=nn.GELU,
            norm_layer=nn.LayerNorm,
    ):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention2(
            dim,
            num_heads=num_heads,
            qkv_bias=qkv_bias,
            attn_drop=attn_drop,
            proj_drop=drop,
        )
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            act_layer=act_layer,
            drop=drop,
        )

    def forward(self, x, target):
        _x = self.attn(x, target)
        x = x + self.drop_path(_x)
        x = x + self.drop_path(self.mlp(self.norm1(x)))
        x = self.norm2(x)
        return x


class PatchEmbed(nn.Module):

    def __init__(self, in_chans=3, embed_dim=768):
        super().__init__()

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(32, 52), stride=(16, 26))

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class VisionTransformer(nn.Module):

    def __init__(
            self,
            in_chans=3,
            embed_dim=768,
            depth=6,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.1,
            attn_drop_rate=0.0,
            drop_path_rate=0.0
    ):

        super().__init__()

        norm_layer = nn.LayerNorm

        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim)

        self.pos_embed = nn.Parameter(torch.zeros(1, 961, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.to_target_embedding = nn.Sequential(
            Rearrange('b c (pn1 h) (pn2 w) -> b (pn1 pn2) (h w c)', pn1=1, pn2=1),
            nn.Linear(6144, embed_dim)
        )

        self.target_self_attention = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )

        self.target_pre_norm = norm_layer(embed_dim)
        self.img_pre_norm = norm_layer(embed_dim)

        # dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block2(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    norm_layer=norm_layer,
                )
                for _ in range(depth)
            ]
        )
        # self.norm = norm_layer(embed_dim)
        #
        # self.apply(self._init_weights)
        self.head = SegHead(in_chans=embed_dim)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, img, target):

        img = self.patch_embed(img)
        print(img.shape)

        img = img + self.pos_embed
        img = self.pos_drop(img)

        target = self.to_target_embedding(target)

        for tsa in self.target_self_attention:
            target = tsa(target)

        x = self.img_pre_norm(img)
        target = self.target_pre_norm(target)

        for blk in self.blocks:
            x = blk(x, target)

        x = rearrange(x, 'b (h w) c -> b c h w', h=31)

        x = self.head(x)

        return x


if __name__ == '__main__':
    model = VisionTransformer()
    img = torch.randn(1, 3, 512, 832)
    target = torch.randn(1, 3, 32, 64)
    start = time.time()
    out = model(img, target)
    print(time.time() - start)
    print(out.shape)
