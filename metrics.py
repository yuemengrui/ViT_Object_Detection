# *_*coding:utf-8 *_*
# @Author : YueMengRui
import numpy as np


class CenterMetric(object):
    def __init__(self, **kwargs):
        self.correct_num = 0
        self.all_num = 0
        self.reset()

    def __call__(self, preds, labels, *args, **kwargs):
        correct_num = 0
        all_num = 0
        for i in range(len(preds)):

            pred_c_x = preds[i][0]
            pred_c_y = preds[i][1]

            label_c_x_min = labels[i][0]
            label_c_x_max = labels[i][1]
            label_c_y_min = labels[i][2]
            label_c_y_max = labels[i][3]

            if label_c_x_min <= pred_c_x <= label_c_x_max and label_c_y_min <= pred_c_y <= label_c_y_max:
                correct_num += 1

            all_num += 1

        self.correct_num += correct_num
        self.all_num += all_num

        return {
            'acc': correct_num / all_num
        }

    def get_metric(self):

        acc = 1.0 * self.correct_num / self.all_num
        self.reset()
        return {'acc': acc}

    def reset(self):
        self.correct_num = 0
        self.all_num = 0
        self.norm_edit_dis = 0


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
        acc = self.Pixel_Accuracy()

        acc_cls = self.Pixel_Accuracy_Class()

        mean_iou = self.Mean_Intersection_over_Union()

        fwavacc = self.Frequency_Weighted_Intersection_over_Union()

        cls_iou = self.Class_IOU()

        return {
            "OverallAcc": acc,
            "MeanAcc": acc_cls,
            "FreqWAcc": fwavacc,
            "MeanIoU": mean_iou,
            "ClassIoU": cls_iou,
        }

    def reset(self):
        self.confusion_matrix = np.zeros((self.n_classes, self.n_classes))
