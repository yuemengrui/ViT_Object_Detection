# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
import torch.nn as nn
from functools import partial
from einops import rearrange
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import time


class SegHead(nn.Module):

    def __init__(self, in_chans, img_size=(512, 1024)):
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

    def forward(self, x, mask=None):
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
        if mask is not None:
            mask = mask.bool()
            attn = attn.masked_fill(~mask[:, None, None, :], float("-inf"))
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn


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

    def forward(self, x, mask=None):
        _x, attn = self.attn(self.norm1(x), mask=mask)
        x = x + self.drop_path(_x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x, attn


class PatchEmbed(nn.Module):
    """ Image to Patch Embedding"""

    def __init__(
            self,
            img_size=(224, 224),
            patch_size=(16, 16),
            in_chans=3,
            embed_dim=768
    ):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(
            in_chans,
            embed_dim,
            kernel_size=patch_size,
            stride=patch_size,
            bias=True,
        )

    def forward(self, x):
        # B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        x = self.proj(x)
        return x


class VisionTransformer(nn.Module):

    def __init__(
            self,
            img_size=(512, 1024),
            patch_size=(16, 32),
            target_size=(32, 64),
            target_patch_size=(32, 64),
            in_chans=3,
            embed_dim=768,
            depth=12,
            num_heads=12,
            mlp_ratio=4.0,
            qkv_bias=True,
            qk_scale=None,
            drop_rate=0.1,
            attn_drop_rate=0.0,
            drop_path_rate=0.0
    ):

        super().__init__()

        norm_layer = partial(nn.LayerNorm, eps=1e-6)

        self.patch_embed = PatchEmbed(
            img_size=img_size,
            patch_size=patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        trunc_normal_(self.pos_embed, std=0.02)
        trunc_normal_(self.cls_token, std=0.02)

        self.target_patch_embed = PatchEmbed(
            img_size=target_size,
            patch_size=target_patch_size,
            in_chans=in_chans,
            embed_dim=embed_dim,
        )
        target_num_patches = self.target_patch_embed.num_patches

        self.target_cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.target_pos_embed = nn.Parameter(torch.zeros(1, target_num_patches + 1, embed_dim))
        trunc_normal_(self.target_pos_embed, std=0.02)
        trunc_normal_(self.target_cls_token, std=0.02)

        self.pos_drop = nn.Dropout(p=drop_rate)

        self.token_type_embeddings = nn.Embedding(2, embed_dim)

        self.pre_norm = norm_layer(embed_dim)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )
        self.norm = norm_layer(embed_dim)

        self.apply(self._init_weights)

        self.head = SegHead(in_chans=embed_dim, img_size=img_size)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=0.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {"pos_embed", "cls_token"}

    def forward(self, img, target):
        B, _, _, _ = img.shape

        img = self.patch_embed(img)
        _, _, h, w = img.shape
        img = img.flatten(2).transpose(1, 2)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        img = torch.cat((cls_tokens, img), dim=1)
        img = img + self.pos_embed
        img = self.pos_drop(img)
        img += self.token_type_embeddings(torch.zeros(B, 1).long().cuda())

        target = self.target_patch_embed(target)
        target = target.flatten(2).transpose(1, 2)
        target_cls_tokens = self.target_cls_token.expand(B, -1, -1)
        target = torch.cat((target_cls_tokens, target), dim=1)
        target = target + self.target_pos_embed
        target = self.pos_drop(target)
        target += self.token_type_embeddings(torch.ones(B, 1).long().cuda())

        x = torch.cat((target, img), dim=1)

        x = self.pre_norm(x)

        for blk in self.blocks:
            x, _ = blk(x)

        x = self.norm(x)

        x = x[:, 3:, :]

        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)

        x = self.head(x)

        return x


if __name__ == '__main__':
    model = VisionTransformer()
    img = torch.randn(1, 3, 512, 1024)
    target = torch.randn(1, 3, 32, 64)
    start = time.time()
    out = model(img, target)
    print(time.time() - start)
    print(out.shape)
