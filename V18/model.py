# *_*coding:utf-8 *_*
# @Author : YueMengRui
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange
from timm.models.layers import trunc_normal_
import time


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


def elu_feature_map(x):
    return F.elu(x) + 1


class LinearAttention(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.feature_map = elu_feature_map
        self.eps = eps

    def forward(self, queries, keys, values, q_mask=None, kv_mask=None):
        """ Multi-Head linear attention proposed in "Transformers are RNNs"
        Args:
            queries: [N, L, H, D]
            keys: [N, S, H, D]
            values: [N, S, H, D]
            q_mask: [N, L]
            kv_mask: [N, S]
        Returns:
            queried_values: (N, L, H, D)
        """
        Q = self.feature_map(queries)
        K = self.feature_map(keys)

        # set padded position to zero
        if q_mask is not None:
            Q = Q * q_mask[:, :, None, None]
        if kv_mask is not None:
            K = K * kv_mask[:, :, None, None]
            values = values * kv_mask[:, :, None, None]

        v_length = values.size(1)
        values = values / v_length  # prevent fp16 overflow
        KV = torch.einsum("nshd,nshv->nhdv", K, values)  # (S,D)' @ S,V
        Z = 1 / (torch.einsum("nlhd,nhd->nlh", Q, K.sum(dim=1)) + self.eps)
        queried_values = torch.einsum("nlhd,nhdv,nlh->nlhv", Q, KV, Z) * v_length

        return queried_values.contiguous()


class LoFTREncoderLayer(nn.Module):
    def __init__(self,
                 d_model,
                 nhead,
                 attention='linear'):
        super(LoFTREncoderLayer, self).__init__()

        self.dim = d_model // nhead
        self.nhead = nhead

        # multi-head attention
        self.q_proj = nn.Linear(d_model, d_model, bias=False)
        self.k_proj = nn.Linear(d_model, d_model, bias=False)
        self.v_proj = nn.Linear(d_model, d_model, bias=False)
        self.attention = LinearAttention()
        self.merge = nn.Linear(d_model, d_model, bias=False)

        # feed-forward network
        self.mlp = nn.Sequential(
            nn.Linear(d_model * 2, d_model * 2, bias=False),
            nn.ReLU(True),
            nn.Linear(d_model * 2, d_model, bias=False),
        )

        # norm and dropout
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, source, x_mask=None, source_mask=None):
        """
        Args:
            x (torch.Tensor): [N, L, C]
            source (torch.Tensor): [N, S, C]
            x_mask (torch.Tensor): [N, L] (optional)
            source_mask (torch.Tensor): [N, S] (optional)
        """
        bs = x.size(0)
        query, key, value = x, source, source

        # multi-head attention
        query = self.q_proj(query).view(bs, -1, self.nhead, self.dim)  # [N, L, (H, D)]
        key = self.k_proj(key).view(bs, -1, self.nhead, self.dim)  # [N, S, (H, D)]
        value = self.v_proj(value).view(bs, -1, self.nhead, self.dim)
        message = self.attention(query, key, value, q_mask=x_mask, kv_mask=source_mask)  # [N, L, (H, D)]
        message = self.merge(message.view(bs, -1, self.nhead * self.dim))  # [N, L, C]
        message = self.norm1(message)

        # feed-forward network
        message = self.mlp(torch.cat([x, message], dim=2))
        message = self.norm2(message)

        return x + message


class LocalFeatureTransformer(nn.Module):
    """A Local Feature Transformer (LoFTR) module."""

    def __init__(self, embed_dim, num_heads, depth):
        super(LocalFeatureTransformer, self).__init__()

        self.d_model = embed_dim
        self.nhead = num_heads
        self.layer_names = ['self', 'cross'] * depth
        encoder_layer = LoFTREncoderLayer(embed_dim, num_heads)
        self.layers = nn.ModuleList([copy.deepcopy(encoder_layer) for _ in range(len(self.layer_names))])
        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def forward(self, feat0, feat1, mask0=None, mask1=None):
        """
        Args:
            feat0 (torch.Tensor): [N, L, C]
            feat1 (torch.Tensor): [N, S, C]
            mask0 (torch.Tensor): [N, L] (optional)
            mask1 (torch.Tensor): [N, S] (optional)
        """

        assert self.d_model == feat0.size(2), "the feature number of src and transformer must be equal"

        for layer, name in zip(self.layers, self.layer_names):
            if name == 'self':
                feat0 = layer(feat0, feat0, mask0, mask0)
                feat1 = layer(feat1, feat1, mask1, mask1)
            elif name == 'cross':
                feat0 = layer(feat0, feat1, mask0, mask1)
                feat1 = layer(feat1, feat0, mask1, mask0)
            else:
                raise KeyError

        return feat0, feat1


class PatchEmbed(nn.Module):

    def __init__(self, in_chans, embed_dim):
        super().__init__()

        # self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(32, 52), stride=(16, 26))
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=(16, 26), stride=(8, 13))

        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x = self.proj(x)  # [N, 768, 31, 31]
        x = x.flatten(2).transpose(1, 2)
        x = self.norm(x)
        return x


class ViT(nn.Module):

    def __init__(self, in_chans=3, embed_dim=512, depth=6, num_heads=8, drop_rate=0.1):
        super().__init__()

        self.patch_embed = PatchEmbed(in_chans=in_chans, embed_dim=embed_dim)
        # self.pos_embed = nn.Parameter(torch.zeros(1, 3969, embed_dim))
        # trunc_normal_(self.pos_embed, std=0.02)

        self.to_target_embedding = nn.Sequential(
            Rearrange('b c (pn1 h) (pn2 w) -> b (pn1 pn2) (h w c)', pn1=1, pn2=1),
            nn.Linear(6144, embed_dim)
        )
        # self.target_pos_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        # trunc_normal_(self.target_pos_embed, std=0.02)

        # self.pos_drop = nn.Dropout(p=drop_rate)

        self.boxes_embedding = nn.Linear(8, embed_dim)
        self.text_embedding = nn.Linear(64, embed_dim)

        self.text_transformer = LocalFeatureTransformer(embed_dim, num_heads, 4)
        self.img_transformer = LocalFeatureTransformer(embed_dim, num_heads, 4)
        self.transformer = LocalFeatureTransformer(embed_dim, num_heads, 4)

        self.mlp_head = MLPHead(embed_dim, embed_dim, 4, 3)

    def forward(self, img, img_boxes, img_texts, target, target_texts):
        img = self.patch_embed(img)  # [N, 3969, 512]
        # img = img + self.pos_embed
        # img = self.pos_drop(img)

        target = self.to_target_embedding(target)  # [N, 1, 512]

        # target = target + self.target_pos_embed
        # target = self.pos_drop(target)

        # boxes_embeding = self.boxes_embedding(img_boxes).unsqueeze(0)

        img_text_embeding = self.text_embedding(img_texts)
        img_text_embeding = F.normalize(img_text_embeding).unsqueeze(0)

        target_text_embeding = self.text_embedding(target_texts)
        target_text_embeding = F.normalize(target_text_embeding).unsqueeze(0)

        target_text, img_text = self.text_transformer(target_text_embeding, img_text_embeding)

        target, img = self.img_transformer(target, img)

        targets = torch.cat((target, target_text), dim=1)
        imgs = torch.cat((img, img_text), dim=1)

        _, x = self.transformer(targets, imgs)

        x = x.mean(dim=1)

        x = self.mlp_head(x).sigmoid()

        return x


if __name__ == '__main__':
    model = ViT()
    img = torch.randn(1, 3, 512, 832)
    target = torch.randn(1, 3, 32, 64)
    start = time.time()
    out = model(img, target)
    print(time.time() - start)
    print(out.shape)

    # net = PatchEmbed()

    # out = net(img)
    # print(out.shape)
