# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
import torch.nn as nn
import torch.nn.functional as F
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
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=kernel_size, stride=1,
                      padding=padding),
            nn.ReLU()
        )

        # self.ln2 = nn.LayerNorm(out_channels)

    def forward(self, input):
        x = self.conv1(input)
        B, C, H, W = x.shape
        x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        x = self.ln1(x)
        x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        x = self.conv2(x)
        # B, C, H, W = x.shape
        # x = x.view(B, C, -1).transpose(-2, -1).contiguous()
        # x = self.ln2(x)
        # x = x.transpose(-2, -1).contiguous().view(B, C, H, W)

        return x


class MatchUNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.img_conv1 = BasicConvBlock(3, 32, 7, 4, 2)
        self.target_conv1 = BasicConvBlock(3, 32, 3, 2)

        self.img_conv2 = BasicConvBlock(32, 64, 3, 2)
        self.target_conv2 = BasicConvBlock(32, 64, 3)

        self.img_conv3 = BasicConvBlock(64, 128, 3)
        self.target_conv3 = BasicConvBlock(64, 128, 3)

        self.img_conv4 = BasicConvBlock(128, 256, 3)
        self.target_conv4 = BasicConvBlock(128, 256, 3)

        self.img_conv5 = BasicConvBlock(256, 512, 3)
        self.target_conv5 = BasicConvBlock(256, 512, 3)

        self.conv_seg = nn.Conv2d(5, 2, 1)

    def forward(self, x, target):
        ori_h, ori_w = x.shape[2:]

        x = self.img_conv1(x)
        print(x.shape)
        target = self.target_conv1(target)
        print(target.shape)
        f1 = F.conv2d(x, target)
        print(f1.shape)

        x = self.img_conv2(x)
        print(x.shape)
        target = self.target_conv2(target)
        print(target.shape)
        f2 = F.conv2d(x, target)
        print(f2.shape)

        x = self.img_conv3(x)
        print(x.shape)
        target = self.target_conv3(target)
        print(target.shape)
        f3 = F.conv2d(x, target)
        print(f3.shape)

        x = self.img_conv4(x)
        print(x.shape)
        target = self.target_conv4(target)
        print(target.shape)
        f4 = F.conv2d(x, target)
        print(f4.shape)

        x = self.img_conv5(x)
        target = self.target_conv5(target)
        f5 = F.conv2d(x, target)
        print(f5.shape)

        ff = torch.cat((f2, f3, f4, f5), dim=1)
        ff = F.interpolate(ff, size=f1.shape[2:], mode='bilinear', align_corners=True)
        ff = torch.cat((f1, ff), dim=1)

        out = self.conv_seg(ff)
        out = F.interpolate(out, size=(ori_h, ori_w), mode='bilinear', align_corners=True)

        return out


if __name__ == '__main__':
    model = MatchUNet()
    img = torch.randn(1, 3, 368, 368)  # min_edge: 372, max_edge:832
    target = torch.randn(1, 3, 96, 96)  # min_edge: 22, max_edge: 96
    start = time.time()
    out = model(img, target)
    print(time.time() - start)

    print(out.shape)

# 246, 406
# 6, 22
