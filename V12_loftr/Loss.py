# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
import torch.nn as nn
import numpy as np


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()

    def forward(self, pred0, pred1, box, label_box):

        box = box[0]
        label_box = label_box[0]

        ps = pred0.shape[0]

        ori_w = box[2] - box[0]
        ori_h = box[3] - box[1]

        label_w = label_box[2] - label_box[0]
        label_h = label_box[3] - label_box[1]

        loss_list = []
        for i in range(ps):
            point_0 = pred0[i]
            point_1 = pred1[i]
            if box[0] <= point_0[0] <= box[2] and box[1] <= point_0[1] <= box[3]:
                x_scale = (point_0[0] - box[0]) / ori_w
                y_scale = (point_0[1] - box[1]) / ori_h

                new_p_x = x_scale * label_w + label_box[0]
                new_p_y = y_scale * label_h + label_box[1]

                loss_list.append(torch.pow(torch.pow(new_p_x - point_1[0], 2) + torch.pow(new_p_y - point_1[1], 2), 0.5))

        if len(loss_list) == 0:
            return torch.tensor(1000, requires_grad=True)

        return torch.mean(torch.tensor(loss_list, requires_grad=True))
