# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
        self.epsilon = 1e-5

    def forward(self, predict, target):
        assert predict.size() == target.size(), "the size of predict and target must be equal."
        num = predict.size(0)

        pre = torch.sigmoid(predict).view(num, -1)
        tar = target.view(num, -1)

        intersection = (pre * tar).sum(-1).sum()  # 利用预测值与标签相乘当作交集
        union = (pre + tar).sum(-1).sum()

        score = 1 - 2 * (intersection + self.epsilon) / (union + self.epsilon)

        return score


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.1, gamma=2.0, use_sigmoid=True):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.use_sigmoid = use_sigmoid
        if use_sigmoid:
            self.sigmoid = nn.Sigmoid()

    def forward(self, pred: torch.Tensor, target: torch.Tensor):
        if self.use_sigmoid:
            pred = self.sigmoid(pred)
        print('pred: ',pred.shape)
        print('label: ',target.shape)
        pred = pred.view(-1)
        print('pred: ',pred.shape)
        label = target.view(-1)
        print('label: ',label.shape)
        pos = torch.nonzero(label > 0).squeeze(1)
        pos_num = max(pos.numel(), 1.0)
        mask = ~(label == -1)
        pred = pred[mask]
        label = label[mask]
        focal_weight = self.alpha * (label - pred).abs().pow(self.gamma) * (label > 0.0).float() + (
                1 - self.alpha) * pred.abs().pow(self.gamma) * (label <= 0.0).float()
        loss = F.binary_cross_entropy(pred, label, reduction='none') * focal_weight
        return loss.sum() / pos_num
