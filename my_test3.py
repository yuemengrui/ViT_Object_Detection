# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import cv2

# class Net(nn.Module):
#
#     def __init__(self):
#         super(Net, self).__init__()
#         # 1个输入，6个输出，5*5的卷积
#         # 内核
#         self.conv1 = nn.Conv2d(1, 6, 3)
#         self.conv2 = nn.Conv2d(6, 16, 3)
#         # 映射函数：线性——y=Wx+b
#         self.fc1 = nn.Linear(16 * 5 * 5, 120)  # 输入特征值：16*5*5，输出特征值：120
#         self.fc2 = nn.Linear(120, 84)
#         self.fc3 = nn.Linear(84, 10)
#
#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(self.conv2(x))
#         x = x.view(-1, self.num_flat_features(x))
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = self.fc3(x)
#         return x
#

if __name__ == '__main__':
    # x = torch.randn(1,2,2)
    # y = torch.randn(1,1,2)
    # print(x)
    # print(y)
    #
    # z = x+y
    # print(z)
    # l = [1280/1024, 800/600, 1440/1050, 1680/1050, 1280/768, 1360/768, 1366/768, 1280/600]
    # print(np.mean(np.array(l)))  # 1.61303
    # for i in range(1):
    #     print(i)
    # label = cv2.imread('/Users/yuemengrui/Data/RPAUI/train_data_new/labels/0afb3af92ca3e735a1aa444ea1535ec9.png', -1)

    # cv2.imshow('l', label * 255)
    # cv2.waitKey(0)

    # b = [10, 10, 30, 10, 30, 20, 10, 20]
    #
    # bb = np.array(b).reshape((4, 2))
    # print(bb)
    # print(min(bb[:, 0]))
    # print(max(bb[:, 0]))
    # print(min(bb[:, 1]))
    # print(max(bb[:, 1]))
    # c = np.array([40, 30])
    # scale = np.array((0.1, 0.5))
    #
    # d = (bb / c * scale).reshape((8,)).tolist()
    # print(d)
    l = [[1,1], [2,2], [3,3]]
    ll = []
    ll.extend(l)
    print(ll)
