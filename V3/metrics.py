# *_*coding:utf-8 *_*
# @Author : YueMengRui
import numpy as np
import logger as logger


def iou(box1, box2):
    """
    两个框（二维）的 iou 计算
    注意：边框以左上为原点
    box:[x1,y1,x2,y2],依次为左上右下坐标
    """

    h = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    w = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    area_box1 = ((box1[2] - box1[0]) * (box1[3] - box1[1]))
    area_box2 = ((box2[2] - box2[0]) * (box2[3] - box2[1]))
    inter = w * h
    union = area_box1 + area_box2 - inter
    iou = inter / union
    return iou


class CornerMetric(object):
    def __init__(self, **kwargs):
        self.correct_num = 0
        self.all_num = 0
        self.reset()

    def __call__(self, preds, labels, *args, **kwargs):
        correct_num = 0
        all_num = 0
        logger.val.info('preds: {}'.format(str(preds)))
        logger.val.info('labels: {}'.format(str(labels)))
        for i in range(len(preds)):

            logger.val.info('pred_{}:{}'.format(i, preds[i]))

            if min(preds[i]) > 0:
                if preds[i][2] > preds[i][0] and preds[i][3] > preds[i][1]:
                    pred_x1 = preds[i][0] * 1024
                    pred_y1 = preds[i][1] * 512
                    pred_x2 = preds[i][2] * 1024
                    pred_y2 = preds[i][3] * 512
                    logger.val.info('pred [{}, {}, {}, {}]'.format(pred_x1, pred_y1, pred_x2, pred_y2))

                    label_x1 = labels[i][0] * 1024
                    label_y1 = labels[i][1] * 512
                    label_x2 = labels[i][2] * 1024
                    label_y2 = labels[i][3] * 512
                    logger.val.info('label [{}, {}, {}, {}]'.format(label_x1, label_y1, label_x2, label_y2))

                    if iou([pred_x1, pred_y1, pred_x2, pred_y2], [label_x1, label_y1, label_x2, label_y2]) > 0.7:
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


class CenterMetric(object):
    def __init__(self, **kwargs):
        self.correct_num = 0
        self.all_num = 0
        self.reset()

    def __call__(self, preds, labels, *args, **kwargs):
        correct_num = 0
        all_num = 0
        # logger.val.info('preds: {}'.format(str(preds)))
        # logger.val.info('labels: {}'.format(str(labels)))
        for i in range(len(preds)):

            pred_c_x = preds[i][0]
            pred_c_y = preds[i][1]
            # logger.val.info('pred_c_x: {} pred_c_y:{}'.format(pred_c_x, pred_c_y))

            label_c_x_min = labels[i][0]
            label_c_x_max = labels[i][1]
            label_c_y_min = labels[i][2]
            label_c_y_max = labels[i][3]
            # logger.val.info('label_c_x_min:{} label_c_x_max:{}'.format(label_c_x_min, label_c_x_max))
            # logger.val.info('label_c_y_min:{} label_c_y_max:{}'.format(label_c_y_min, label_c_y_max))

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
