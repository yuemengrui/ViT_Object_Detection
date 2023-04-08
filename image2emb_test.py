# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops.layers.torch import Rearrange


class Model(nn.Module):

    def __init__(self, patch_size=(32, 64), model_dim=512, c=3):
        super().__init__()

        self.patch_size = patch_size
        self.patch_depth = patch_size[0] * patch_size[1] * c
        self.patch_weight = nn.Parameter(torch.randn(self.patch_depth, model_dim))
        self.kernel = self.patch_weight.transpose(0, 1).reshape((-1, c, patch_size[0], patch_size[1]))

        self.patch_emb = nn.Unfold(kernel_size=patch_size, stride=patch_size)

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)', p1=32, p2=64),
            nn.Linear(6144, 512)
        )

    def forward(self, x):
        # x = self.patch_emb(x).transpose(-1, -2)
        # x = x @ self.patch_weight
        x = self.to_patch_embedding(x)
        return x

    def image2emb_naive(self, image):
        patch = F.unfold(image, kernel_size=self.patch_size, stride=self.patch_size)
        patch = patch.transpose(-1, -2)
        patch_embedding = patch @ self.patch_weight
        return patch_embedding

    def image2emb_conv(self, image):
        conv_output = F.conv2d(image, self.kernel, stride=self.patch_size)
        n, c, h, w = conv_output.shape
        patch_embedding = conv_output.reshape((n, c, h * w)).transpose(-1, -2)
        return patch_embedding


if __name__ == '__main__':
    model = Model()
    image = torch.randn(1, 3, 512, 1024)

    out = model(image)
    print(out.shape)
    print(out)

    out_naive = model.image2emb_naive(image)
    print(out_naive.shape)
    print(out_naive)

    out_conv = model.image2emb_conv(image)
    print(out_conv.shape)
    print(out_conv)

# def image2emb_naive(image, patch_size, weight):
#     patch = F.unfold(image, kernel_size=patch_size, stride=patch_size).transpose(-1, -2)
#     patch_embedding = patch @ weight
#     return patch_embedding
#
#
# def image2emb_conv(image, kernel, stride):
#     conv_output = F.conv2d(image, kernel, stride=stride)
#     n, c, h, w = conv_output.shape
#     patch_embedding = conv_output.reshape((n, c, h * w)).transpose(-1, -2)
#     return patch_embedding


# n, c, h, w = 1, 3, 8, 8
# patch_size = 4
# model_dim = 8
#
# patch_depth = patch_size * patch_size * c
#
# image = torch.randn((n, c, h, w))
#
# weight = torch.randn((patch_depth, model_dim))
#
# kernel = weight.transpose(0, 1).reshape((-1, c, patch_size, patch_size))
#
# patch_emb_naive = image2emb_naive(image, patch_size, weight)
#
# patch_emb_conv = image2emb_conv(image, kernel, stride=patch_size)
#
# print(patch_emb_naive)
# print(patch_emb_conv)
