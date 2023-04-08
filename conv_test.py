# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
import torch.nn.functional as F

img = torch.randn(1, 3, 512, 1024)
target = torch.randn(1, 3, 32, 64)
# target1 = torch.randn(1, 3, 32, 64)

# conv = torch.nn.Conv2d(3, 1, kernel_size=(32, 64), padding=0, bias=False)
# conv.weight.data = torch.randn(3, 3, 32, 64)
# out = conv(img)
# print(out.shape)
# # print(out)
# o = conv(target)
# print(o.shape)
#
# x = out * o
# print(x.shape)
out = F.conv2d(target, target, stride=1, padding=0)
print(out.shape)
print(out)
# print(type(out))
#
# out = out.squeeze(dim=0)
# print(out.shape)
# b = img.size()[0]
# print(b)

# for i in range(b):
#     im = img[i, :, :, :]
#     print(im.shape)
