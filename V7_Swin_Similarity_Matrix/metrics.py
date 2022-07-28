# *_*coding:utf-8 *_*
# @Author : YueMengRui
import numpy as np


class SegMetrics(object):

    def __init__(self, n_classes=2):
        self.n_classes = n_classes
        self.confusion_matrix = np.zeros((n_classes, n_classes))

    def update(self, label_trues, label_preds):
        for lt, lp in zip(label_trues, label_preds):
            self.confusion_matrix += self._fast_hist(lt.flatten(), lp.flatten())

    def _fast_hist(self, label_true, label_pred):
        mask = (label_true >= 0) & (label_true < self.n_classes)
        hist = np.bincount(
            self.n_classes * label_true[mask].astype(int) + label_pred[mask],
            minlength=self.n_classes ** 2,
        ).reshape(self.n_classes, self.n_classes)
        return hist

    def Pixel_Accuracy(self):
        Acc = np.diag(self.confusion_matrix).sum() / self.confusion_matrix.sum()
        return Acc

    def Pixel_Accuracy_Class(self):
        Acc = np.diag(self.confusion_matrix) / self.confusion_matrix.sum(axis=1)
        Acc = np.nanmean(Acc)
        return Acc

    def Mean_Intersection_over_Union(self):
        MIoU = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))
        MIoU = np.nanmean(MIoU)
        return MIoU

    def Frequency_Weighted_Intersection_over_Union(self):
        freq = np.sum(self.confusion_matrix, axis=1) / np.sum(self.confusion_matrix)
        iu = np.diag(self.confusion_matrix) / (
                np.sum(self.confusion_matrix, axis=1) + np.sum(self.confusion_matrix, axis=0) -
                np.diag(self.confusion_matrix))

        FWIoU = (freq[freq > 0] * iu[freq > 0]).sum()
        return FWIoU

    def Class_IOU(self):
        iou = np.diag(self.confusion_matrix) / (
                self.confusion_matrix.sum(axis=1) + self.confusion_matrix.sum(axis=0) - np.diag(
            self.confusion_matrix))

        cls_iou = dict(zip(range(self.n_classes), iou))

        return cls_iou

    def get_results(self):
        # acc = self.Pixel_Accuracy()

        # acc_cls = self.Pixel_Accuracy_Class()

        mean_iou = self.Mean_Intersection_over_Union()

        # fwavacc = self.Frequency_Weighted_Intersection_over_Union()

        cls_iou = self.Class_IOU()

        return {
            # "OverallAcc": acc,
            # "MeanAcc": acc_cls,
            # "FreqWAcc": fwavacc,
            "MeanIoU": mean_iou,
            "ClassIoU": cls_iou,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
