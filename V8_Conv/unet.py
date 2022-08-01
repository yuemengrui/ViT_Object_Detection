# *_*coding:utf-8 *_*
# @Author : yuemengrui
# @Time : 2021-05-26 下午4:40
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class BasicConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(BasicConvBlock, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=kernel_size, stride=stride,
                      padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(in_channels=out_channels, out_channels=out_channels, kernel_size=(kernel_size[1], kernel_size[0]),
                      stride=stride,
                      padding=(padding[1], padding[0])),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )

    def forward(self, input):
        return self.conv(input)


class UNet(nn.Module):
    def __init__(self, num_classes=3):
        super(UNet, self).__init__()
        self.down_sample1 = BasicConvBlock(3, 64, (7, 3), 1, (1, 3))
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.down_sample2 = BasicConvBlock(64, 128, (3, 7), 1, (3, 1))
        self.down_sample3 = BasicConvBlock(128, 256, (7, 3), 1, (1, 3))
        self.down_sample4 = BasicConvBlock(256, 512, (3, 7), 1, (3, 1))
        self.conv_mid = BasicConvBlock(512, 1024, (3, 3), 1, (1, 1))
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.conv_1_1 = nn.Conv2d(1536, 1024, 1, 1, 0)
        self.up_sample1 = BasicConvBlock(1024, 512, (7, 3), 1, (1, 3))
        self.conv_1_2 = nn.Conv2d(768, 512, 1, 1, 0)
        self.up_sample2 = BasicConvBlock(512, 256, (3, 7), 1, (3, 1))
        self.conv_1_3 = nn.Conv2d(384, 256, 1, 1, 0)
        self.up_sample3 = BasicConvBlock(256, 128, (7, 3), 1, (1, 3))
        self.conv_1_4 = nn.Conv2d(192, 128, 1, 1, 0)
        self.up_sample4 = BasicConvBlock(128, 64, (3, 7), 1, (3, 1))
        self.out = nn.Conv2d(64, num_classes, 1, 1, 0)

    def forward(self, input):
        down_sample1 = self.down_sample1(input)
        print('down_sample1: ', down_sample1.shape)

        x = self.pool(down_sample1)
        down_sample2 = self.down_sample2(x)
        print('down_sample2: ', down_sample2.shape)

        x = self.pool(down_sample2)
        down_sample3 = self.down_sample3(x)
        print('down_sample3: ', down_sample3.shape)

        x = self.pool(down_sample3)
        down_sample4 = self.down_sample4(x)
        print('down_sample4: ', down_sample4.shape)

        x = self.pool(down_sample4)
        x = self.conv_mid(x)
        print('mid: ', x.shape)

        x = self.up(x)
        print('up1: ', x.shape)
        x = torch.cat((x, down_sample4), dim=1)
        print('11: ', x.shape)
        x = self.conv_1_1(x)
        x = self.up_sample1(x)
        print('up_sample1: ', x.shape)

        x = self.up(x)
        x = torch.cat((x, down_sample3), dim=1)
        x = self.conv_1_2(x)
        x = self.up_sample2(x)

        x = self.up(x)
        x = torch.cat((x, down_sample2), dim=1)
        x = self.conv_1_3(x)
        x = self.up_sample3(x)

        x = self.up(x)
        x = torch.cat((x, down_sample1), dim=1)
        x = self.conv_1_4(x)
        x = self.up_sample4(x)
        x = self.out(x)
        return x


if __name__ == '__main__':
    unet = UNet()
    input = torch.rand((1, 3, 64, 64))

    out = unet(input)
    print(out.shape)
