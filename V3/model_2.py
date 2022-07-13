# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
from torch import nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def conv(in_planes, out_planes, kernel_size=3, stride=1, padding=1, dilation=1):
    return nn.Sequential(
        nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride,
                  padding=padding, dilation=dilation, bias=True),
        nn.BatchNorm2d(out_planes),
        nn.ReLU(inplace=True))


class CornerHead(nn.Module):

    def __init__(self, inplanes=3, channel=32):
        super().__init__()
        '''top-left corner'''
        self.conv1_tl = conv(inplanes, channel)
        self.conv2_tl = conv(channel, channel // 2)
        self.conv3_tl = conv(channel // 2, channel // 4)
        self.conv4_tl = conv(channel // 4, channel // 8)
        self.conv5_tl = nn.Conv2d(channel // 8, 1, kernel_size=1)
        self.fc_tl = nn.Linear(512*1024, 2)

        '''bottom-right corner'''
        self.conv1_br = conv(inplanes, channel)
        self.conv2_br = conv(channel, channel // 2)
        self.conv3_br = conv(channel // 2, channel // 4)
        self.conv4_br = conv(channel // 4, channel // 8)
        self.conv5_br = nn.Conv2d(channel // 8, 1, kernel_size=1)
        self.fc_br = nn.Linear(512*1024, 2)

    def forward(self, x):
        tl, br = self.get_score_map(x)
        return torch.cat((tl, br), dim=1)

    def get_score_map(self, x):
        # top-left branch
        x_tl1 = self.conv1_tl(x)
        x_tl2 = self.conv2_tl(x_tl1)
        x_tl3 = self.conv3_tl(x_tl2)
        x_tl4 = self.conv4_tl(x_tl3)
        tl = self.conv5_tl(x_tl4)

        b = tl.shape[0]
        tl = tl.view(b, -1)

        tl = self.fc_tl(tl)

        # bottom-right branch
        x_br1 = self.conv1_br(x)
        x_br2 = self.conv2_br(x_br1)
        x_br3 = self.conv3_br(x_br2)
        x_br4 = self.conv4_br(x_br3)
        br = self.conv5_br(x_br4)
        br = br.view(b, -1)
        br = self.fc_br(br)
        return tl, br


class MatchHead(nn.Module):

    def __init__(self):
        super().__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv = nn.Conv2d(1, 1, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.upsample(x)
        x = self.conv(x)
        return x


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


class ViT(nn.Module):

    def __init__(self, image_size=(512, 1024), patch_size=(32, 64), dim=512, depth=6, heads=8, mlp_dim=512, channels=3,
                 dim_head=64, dropout=0., emb_dropout=0., pool='cls'):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)

        patch_dim = channels * patch_height * patch_width

        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=patch_height, p2=patch_width),
            nn.Linear(patch_dim, dim)
        )

        self.to_target_embedding = nn.Sequential(
            Rearrange('b c (pn1 h) (pn2 w) -> b (pn1 pn2) (h w c)', pn1=1, pn2=1),
            nn.Linear(patch_dim, dim)
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        # self.pos_embedding = nn.Parameter(torch.randn(1, num_patches, dim))

        # self.cls_token = nn.Parameter(torch.randn(1, 1, dim))

        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool

        self.l = nn.Linear(257, 256)

        self.to_up = nn.Sequential(
            nn.Linear(dim, patch_dim),
            Rearrange('b (h w) (p1 p2 c) -> b c (h p1) (w p2)', p1=patch_height, p2=patch_width, c=3, h=16)
        )

        self.corner_head = CornerHead()

        # self.mlp_head = MLPHead(dim, dim, 2, 3)
        # self.match_head = MatchHead()

    def forward(self, img, target_img=None):
        x = self.to_patch_embedding(img)

        target = self.to_target_embedding(target_img)

        b, n, _ = x.shape
        # cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)
        #
        x = torch.cat((target, x), dim=1)

        #
        # target = torch.cat((cls_tokens, target), dim=1)

        x += self.pos_embedding[:, :(n + 1)]

        # x = self.dropout(x)
        #
        x = self.transformer(x)  # [1, 257, 512]

        x = self.l(x.permute(0, 2, 1)).permute(0, 2, 1)  # [1, 256, 512]

        x = self.to_up(x)  # [1, 3, 512, 1024]

        # x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # x = self.mlp_head(x).sigmoid()  # [x,y,h,w,score]
        # x = x.unsqueeze(1)
        # print('111: ', x.shape)
        # x = self.match_head(x)
        x = self.corner_head(x)

        return x


if __name__ == '__main__':
    # image 1920 1080 -> 960 512
    # patch              60   32
    # patch_num          16   16
    model = ViT(patch_size=(32, 64), dim=512, depth=6, heads=8, mlp_dim=512)

    img = torch.randn((2, 3, 512, 1024))

    target_img = torch.randn((2, 3, 32, 64))

    out = model(img, target_img)
    print(out.shape)
    print(out)

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
