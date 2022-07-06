# *_*coding:utf-8 *_*
# @Author : YueMengRui


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
