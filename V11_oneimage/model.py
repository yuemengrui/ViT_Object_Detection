# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


class MatchHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(3, 2, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.conv(x)
        return x


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


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

    def forward(self, x):
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

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h=self.heads), qkv)

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

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class ViT(nn.Module):

    def __init__(self, image_size=(512, 832), patch_size=(32, 52), dim=512, depth=6, heads=8, mlp_dim=512, channels=3,
                 dim_head=64, dropout=0., emb_dropout=0., pool='cls'):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )

        self.to_target_embedding = nn.Sequential(
            Rearrange('b c (pn1 h) (pn2 w) -> b (pn1 pn2) (h w c)', pn1=1, pn2=1),
            nn.Linear(6144, dim)
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

        self.l = nn.Linear(257, 256)

        self.to_up = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_height, p2=patch_width, c=3, h=16)
        )

        # self.corner_head = CornerHead()

        # self.mlp_head = MLPHead(dim, dim, 2, 3)
        self.match_head = MatchHead()

    def forward(self, img, target_img=None):
        x = self.to_patch_embedding(img)

        target = self.to_target_embedding(target_img)

        b, n, _ = x.shape
        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        #

        #
        # target = torch.cat((cls_tokens, target), dim=1)

        x += self.pos_embedding

        x = torch.cat((target, x), dim=1)

        # x = self.dropout(x)
        #
        x = self.transformer(x)  # [1, 257, 512]

        x = self.l(x.permute(0, 2, 1)).permute(0, 2, 1)  # [1, 256, 512]

        x = self.to_up(x)  # [1, 3, 512, 1024]

        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # x = self.mlp_head(x).sigmoid()  # [x,y,h,w,score]
        # x = x.unsqueeze(1)
        # print('111: ', x.shape)
        x = self.match_head(x)
        # x = self.corner_head(x)

        return x


if __name__ == '__main__':
    import time

    model = ViT()

    img = torch.randn((1, 3, 512, 832))

    target_img = torch.randn((1, 3, 32, 64))
    s = time.time()
    out = model(img, target_img)
    print(time.time() - s)
    print(out.shape)
    # print(out)

    # corner = CornerHead()

    # out = corner(img)
    # print(out)

    # x = torch.max(out1)

    # y = torch.max(out2)

    # print(x)
    # print(y)
    #
    # print(0.1 < x < 0.6)
    # print(0.1 < y < 0.6)
