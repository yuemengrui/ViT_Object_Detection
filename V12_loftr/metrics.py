# *_*coding:utf-8 *_*
# @Author : YueMengRui
import numpy as np
import torch


class Metrics(object):

    def __init__(self):
        self.acc_list = []

    def update(self, pred0, pred1, box, label_box):
        box = box[0]
        label_box = label_box[0]

        ps = pred0.shape[0]
        total = 0
        correct = 0
        for i in range(ps):
            point_0 = pred0[i]
            point_1 = pred1[i]
            if box[0] <= point_0[0] <= box[2] and box[1] <= point_0[1] <= box[3]:
                total += 1

                if label_box[0] <= point_1[0] <= label_box[2] and label_box[1] <= point_1[1] <= label_box[3]:
                    correct += 1
        if total == 0:
            self.acc_list.append(0)
        else:
            self.acc_list.append(correct / total)

    def get_results(self):

        return {'acc': torch.mean(torch.tensor(self.acc_list))}

    def reset(self):
        self.acc_list = []
