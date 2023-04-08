# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import random
import numpy as np


class MLP(nn.Module):

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


if __name__ == '__main__':
    # hidden_dim = 768
    # bbox_embed = MLP(hidden_dim, hidden_dim, 4, 3)
    # preds = torch.randn((2, 2, 8, 8))
    # pred = np.argmax(preds, axis=1)
    # print(pred.shape)
    # print(pred)

    # input = torch.randn((2, 1, 4))
    # print(input)
    # out = input.mean(dim=1)
    # print(out.shape)
    # print(out)
    # out = out.sigmoid()
    # print(out)

    # out = bbox_embed(input)
    #
    # print(out.shape)  # [1, 100, 4]
    # print(out)

    # pred = torch.randn((1, 4)).sigmoid()
    # print(pred)
    # gt = torch.randn((1, 4)).sigmoid()
    # print(gt)
    #
    # loss = F.l1_loss(pred, gt)
    # print(loss)

    # print(random.uniform(1.3, 1.6))
    # x = [0.1, 0.5]
    # index = x.index(max(x))
    # print(index)
    a = 1
    a_1 = 0.9
    b = 1
    b_1 = 1.1

    scale = a/a_1 if a/a_1 != 1 else b/b_1
    print(scale)
