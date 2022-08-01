# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange
import numpy as np
import time


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super(BasicConvBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU()
        )

        self.ln1 = nn.LayerNorm(out_channels)

        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.ReLU()
        )

        self.ln2 = nn.LayerNorm(out_channels)

    def forward(self, input):
        x = self.conv1(input)
        print(x.shape)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        print(x.shape)
        x = self.ln1(x)

        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        print(x.shape)

        x = self.conv2(x)

        B, C, H, W = x.shape
        print(x.shape)

        x = x.view(B, C, -1).transpose(-2, -1).contiguous()

        x = self.ln2(x)

        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        print(x.shape)

        return x


class Merge_Block(nn.Module):
    def __init__(self, dim, dim_out, norm_layer=nn.LayerNorm):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim_out, 3, 2, 1)
        self.norm = norm_layer(dim_out)

    def forward(self, x, H, W):
        B, new_HW, C = x.shape
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)
        x = self.conv(x)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.norm(x)

        return x, H, W


class MatchNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.img_conv = BasicConvBlock(3, 2)

    def forward(self, x):
        x = self.img_conv(x)
        return x


if __name__ == '__main__':
    model = BasicConvBlock(3, 8, 5)
    img = torch.randn(1, 3, 512, 832)
    target = torch.randn(1, 3, 32, 64)
    start = time.time()
    out = model(img)
    print(time.time() - start)

    # print(out.shape)

# 246, 406
# 6, 22
