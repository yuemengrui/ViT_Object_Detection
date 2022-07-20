# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import numpy as np
from torch.utils.data import Dataset
import cv2
import json
import torch
import random
from torchvision import transforms
from torch.utils.data import DataLoader


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


def crop_ok(target_box, binary, threshold=0.6):
    target_bin = binary[target_box[1]:target_box[3], target_box[0]:target_box[2]]

    t_h, t_w = target_bin.shape[:2]

    if np.sum(target_bin == 1) / (t_h * t_w) > threshold:
        return True

    return False


class ImagePadding(object):

    def __init__(self, rate=0.5):
        """
        :param rate: rate = h / w
        """
        self.rate = rate

    def __call__(self, img):
        ori_h, ori_w = img.shape[:2]

        if ori_h / ori_w > 0.5:
            new_w = ori_h * 2
            img = cv2.copyMakeBorder(img, 0, 0, 0, new_w - ori_w, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        elif ori_h / ori_w < 0.5:
            new_h = int(ori_w / 2)
            img = cv2.copyMakeBorder(img, 0, new_h - ori_h, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

        return img


class ImageResize(object):

    def __init__(self, size=(1024, 512)):
        self.size = size

    def __call__(self, img, **kwargs):
        h = img.shape[0]
        scale = self.size[1] / h
        img = cv2.resize(img, self.size)

        return img, scale


class ViTSegDataset(Dataset):

    def __init__(self, dataset_dir, mode='train', target_w_range=(30, 1000), target_h_range=(20, 350),
                 target_w_h_rate=(0.7, 20), threshold=0.6, **kwargs):

        self.target_w_range = target_w_range
        self.target_h_range = target_h_range
        self.target_w_h_rate = target_w_h_rate
        self.threshold = threshold

        self.image_padding = ImagePadding()
        self.image_resize = ImageResize()
        self.target_resize = ImageResize(size=(64, 32))

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                             ])

        self.data_list = self._load_data(dataset_dir, mode)

    def _load_data(self, dataset_dir, mode='train'):
        img_dir = os.path.join(dataset_dir, 'images')
        labels_dir = os.path.join(dataset_dir, 'labels')

        file_list = os.listdir(img_dir)

        data_list = []
        for fil in file_list:
            file_name = fil.split('.')[0]
            img_path = os.path.join(img_dir, fil)
            label_path = os.path.join(labels_dir, file_name + '.png')

            data_list.append({'img_path': img_path, 'label_path': label_path})

        if mode == 'val':
            random.shuffle(data_list)
            return data_list[:50]
        return data_list

    def get_crop_img(self, ori_h, ori_w, binary):
        while True:
            x1 = random.randint(0, ori_w - self.target_w_range[0])
            y1 = random.randint(0, ori_h - self.target_h_range[0])
            target_w = random.randint(self.target_w_range[0], self.target_w_range[1])
            target_h = random.randint(self.target_h_range[0], self.target_h_range[1])

            x2 = min(ori_w, x1 + target_w)
            y2 = min(ori_h, y1 + target_h)

            w = x2 - x1
            h = y2 - y1
            rate = w / h
            if self.target_w_h_rate[0] <= rate <= self.target_w_h_rate[1]:
                if crop_ok([x1, y1, x2, y2], binary, self.threshold):
                    return [x1, y1, x2, y2]

    def __getitem__(self, idx):
        try:
            data = self.data_list[idx]

            img = cv2.imread(data['img_path'])
            binary = cv2.imread(data['label_path'], -1)

            ori_h, ori_w = img.shape[:2]

            label = np.uint8(np.zeros((ori_h, ori_w)))

            target_box = self.get_crop_img(ori_h, ori_w, binary)

            target = img[target_box[1]:target_box[3], target_box[0]:target_box[2]]

            label[target_box[1]:target_box[3], target_box[0]:target_box[2]] = 1

            img = self.image_padding(img)
            img, _ = self.image_resize(img)
            img = self.transform(img)

            target = self.image_padding(target)
            target, _ = self.target_resize(target)
            target = self.transform(target)

            label = self.image_padding(label)
            label, _ = self.image_resize(label)

            label = torch.from_numpy(label)

            return img, target, label

        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(self.__len__()))

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    dataset = ViTSegDataset(dataset_dir='/Users/yuemengrui/Data/RPAUI/train_data')
    for i in range(20):
        img, box = dataset[i]

        cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        cv2.imshow('xx', img)
        cv2.waitKey(0)
#     rate = 0
#     start = time.time()
#     for i in range(100):
#         img, target, label = dataset[i]
#     print(time.time()-start)
#     #     dataset = ViTDetDataset(dataset_dir='/Users/yuemengrui/Data/RPAUI/web_cv_data')
#     #     train_loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)
#
#     # for img, target, label in train_loader:
#     #     print(img.shape)
#     #     print(target.shape)
#     #     print(label.shape)
#     #     break
#     #         print(label)
#     #         print(center_range.numpy().tolist())
#     #         center_range = center_range.numpy().tolist()
#     #         preds = [[0.508238673210144, 0.4653865396976471, 0.49629437923431396], [0.2,0.2,0.2]]
#     #
#     #         for i in range(len(preds)):
#     #             print("==========================")
#     #             print(preds[i])
#     #             print(center_range[i])
#     #
#     #             pred_c_x = preds[i][0]
#     #             pred_c_y = preds[i][1]
#     #
#     #             print(pred_c_x, pred_c_y)
#     #
#     #             label_c_x_min = center_range[i][0]
#     #             label_c_x_max = center_range[i][1]
#     #             label_c_y_min = center_range[i][2]
#     #             label_c_y_max = center_range[i][3]
#     #
#     #             print(label_c_x_min, label_c_x_max)
#     #             print(label_c_y_min, label_c_y_max)
#     #
#     #
