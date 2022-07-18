# *_*coding:utf-8 *_*
# @Author : YueMengRui
import numpy as np
import logger as logger


class ClassifyMetric(object):
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

            pred = preds[i]
            logger.val.info('pred: {} '.format(pred))
            index = pred.index(max(pred))
            logger.val.info('index: {} '.format(index))

            label = labels[i]
            logger.val.info('label: {} '.format(label))

            if index == label:
                logger.val.info('correct')
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
