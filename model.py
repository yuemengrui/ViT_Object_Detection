# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, target=None, **kwargs):
        return self.fn(self.norm(x), target=target, **kwargs)


class MLPHead(nn.Module):

    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.ln = nn.LayerNorm(input_dim)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        x = self.ln(x)
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, **kwargs):
        return self.net(x)


class Attention(nn.Module):
    def __init__(self, dim, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim=-1)
        self.dropout = nn.Dropout(dropout)

        # self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)
        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_k = nn.Linear(dim, inner_dim, bias=False)
        self.to_v = nn.Linear(dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x, target):
        # qkv = self.to_qkv(x).chunk(3, dim=-1)
        # q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        q = rearrange(self.to_q(target), 'b n (h d) -> b h n d', h=self.heads)
        k = rearrange(self.to_k(x), 'b n (h d) -> b h n d', h=self.heads)
        v = rearrange(self.to_v(x), 'b n (h d) -> b h n d', h=self.heads)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout=0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout=dropout))
            ]))

    def forward(self, x, target):
        for attn, ff in self.layers:
            x = attn(x, target) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):

    def __init__(self, image_size=(512, 1024), patch_size=(32, 64), dim=768, depth=12, heads=12, mlp_dim=768, pool='mean',
                 channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.to_target_embedding = nn.Sequential(
            Rearrange('b c (pn1 h) (pn2 w) -> b (pn1 pn2) (h w c)', pn1=16, pn2=16),
            nn.Linear(24, dim),
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

        self.mlp_head = MLPHead(dim, dim, 2, 3)

    def forward(self, img, target_img=None):
        x = self.to_patch_embedding(img)

        target = self.to_target_embedding(target_img)
        b, n, _ = x.shape
        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        #
        # x = torch.cat((cls_tokens, x), dim=1)
        #
        # target = torch.cat((cls_tokens, target), dim=1)

        x += self.pos_embedding[:, :(n + 1)]

        x = self.dropout(x)

        x = self.transformer(x, target)

        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]
        x = x.mean(dim=1)

        x = self.mlp_head(x)  # [x,y,h,w,score]

        return x.sigmoid()


# if __name__ == '__main__':
#     # image 1920 1080 -> 960 512
#     # patch              60   32
#     # patch_num          16   16
#     model = ViT(image_size=(512, 1024), patch_size=(32, 64), dim=768, depth=12, heads=12, mlp_dim=768)
#
#     img = torch.randn((1, 3, 512, 1024))
#
#     target_img = torch.randn((1, 3, 32, 64))
#
#     out = model(img, target_img)
#     print(out.shape)
#     print(out.data.cpu().numpy().tolist())
