# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


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


class MatchNet(nn.Module):

    def __init__(self, dim=544, depth=6, heads=8, mlp_dim=544, dim_head=64, dropout=0., emb_dropout=0.):
        super().__init__()

        self.to_image_embedding = nn.Sequential(
            Rearrange('b c (pn1 h) (pn2 w) -> b (pn1 pn2) (h w c)', pn1=1, pn2=1),
            nn.Linear(6144, 512)
        )

        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.mlp_head = MLPHead(dim, dim, 2, 3)
        # self.soft_max = nn.Softmax(dim=1)

    def forward(self, img, img_text_encode, target_img, target_text_encode):
        x = self.to_image_embedding(img)  # [N,1,512]
        img_text_encode = img_text_encode.unsqueeze(dim=1)  # [N,32]
        x = torch.cat((x, img_text_encode), dim=2)  # [N, 1, 544]

        target = self.to_image_embedding(target_img)  # [N,1,512]
        target_text_encode = target_text_encode.unsqueeze(dim=1)  # [N, 32]
        y = torch.cat((target, target_text_encode), dim=2)  # [N, 1, 544]

        input = torch.cat((x, y), dim=1)  # [N, 2, 544]

        out = self.transformer(input)  # [N, 2, 544]

        out = out.mean(dim=1)  # [N, 544]

        out = self.mlp_head(out)  # [N, 2]

        return out


if __name__ == '__main__':
    model = MatchNet(dim=544, depth=6, heads=8, mlp_dim=544)

    img = torch.randn((1, 3, 32, 64))
    img_text_encode = torch.randn((1, 32))

    target_img = torch.randn((1, 3, 32, 64))
    target_text_encode = torch.randn((1, 32))

    out = model(img, img_text_encode, target_img, target_text_encode)
    print(out.shape)
    print(out)
    # if out.cpu().data.max(1)[1][0] == 1:
    #
    #     score = torch.softmax(out, dim=1)
    #     print(score)
    #     print(score[0][1])
    #     print(float(score[0][1]))
    #     print(score[0][1] > 0.5)
    # print(out.data.max(1)[1][0] == 1)
    # pred = out.data.cpu().numpy().tolist()
    # print(type(pred))
    # print(pred)
