# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import random
import time
import json
from torchvision import transforms
from torch.utils.data import DataLoader


class ImageRandomScaleResize(object):

    def __init__(self):
        self.w_h_scale = [1.25, 1.33, 1.37, 1.6, 1.667, 1.778, 2.13]
        self.screen_scale = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    def __call__(self, img, label):
        h, w = img.shape[:2]

        w_h_scale = random.choice(self.w_h_scale)

        if w / h > w_h_scale:
            new_w = w
            new_h = w / w_h_scale

        elif w / h < w_h_scale:
            new_w = h * w_h_scale
            new_h = h
        else:
            new_w = w
            new_h = h

        screen_scale = random.choice(self.screen_scale)

        new_w = int(new_w * screen_scale)
        new_h = int(new_h * screen_scale)

        img = cv2.resize(img, (new_w, new_h))
        label = cv2.resize(label, (new_w, new_h))

        return img, label


class ImageResize(object):

    def __init__(self, size_range):
        self.size_range = size_range

    def __call__(self, img, **kwargs):
        img = cv2.resize(img, (self.size_range[1], self.size_range[0]))

        return img


class ViTSegDataset(Dataset):

    def __init__(self, dataset_dir, mode='train', **kwargs):

        self.image_resize = ImageResize(size_range=(512, 832))
        self.target_resize = ImageResize(size_range=(32, 64))
        self.image_random_scale_resize = ImageRandomScaleResize()

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.866, 0.873, 0.895), (0.278, 0.254, 0.224)),
                                             ])

        self.data_list = self._load_data(dataset_dir)

    def _load_data(self, dataset_dir):
        img_dir = os.path.join(dataset_dir, 'images')
        labels_dir = os.path.join(dataset_dir, 'prebox_labels')

        file_list = os.listdir(img_dir)

        data_list = []
        for fil in file_list:
            file_name = fil.split('.')[0]
            img_path = os.path.join(img_dir, fil)
            label_path = os.path.join(labels_dir, file_name + '.json')

            with open(label_path, 'r') as f:
                boxes = json.load(f)

            data_list.append({'img_path': img_path, 'boxes': boxes})

        return data_list

    def __getitem__(self, idx):
        try:
            data = self.data_list[idx]

            img = cv2.imread(data['img_path'])
            boxes = data['boxes']

            ori_h, ori_w = img.shape[:2]

            label = np.uint8(np.zeros((ori_h, ori_w)))

            target_box = boxes[random.randint(0, len(boxes) - 1)]

            target = img[target_box[1]:target_box[3], target_box[0]:target_box[2]]

            label[target_box[1]:target_box[3], target_box[0]:target_box[2]] = 1

            img, label = self.image_random_scale_resize(img, label)

            img = self.image_resize(img)

            target = self.target_resize(target)

            label = self.image_resize(label)

            # img = self.transform(img)
            # target = self.transform(target)
            # label = torch.from_numpy(label)

            return img, target, label

        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(self.__len__()))

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    dataset = ViTSegDataset(dataset_dir='/Users/yuemengrui/Data/RPAUI/train_data_prebox')
    # for _ in range(10):
    #     img, target, label = dataset[0]

    num_0 = 0
    num_1 = 0
    for i in range(367):
        print(i)
        #     # s = time.time()
        img, target, label = dataset[i]

        num_1 += np.sum(label == 1)
        num_0 += np.sum(label == 0)

    num_0 = num_0 / 367
    num_1 = num_1 / 367
    print(num_0, num_1, num_0 / num_1)
    # 420971.479 5012.520 83.983
    # 421607.057 4376.942 96.324
    # 421209.283 4774.716 88.216
    # 421361.051 4622.948 91.145
    # 421553.125 4430.874 95.139
    # 421289.365 4694.634 89.738
    # 421239.544 4744.455 88.785
    # 421257.771 4726.228 89.131
    # 421301.029 4682.970 89.964
    # 421092.057 4891.942 86.078

    #                     89.8503
    # print(time.time() - s)
    #
    # cv2.imshow('xx', img)
    # cv2.waitKey(0)
    #
    # cv2.imshow('t', target)
    # cv2.waitKey(0)
    #
    # cv2.imshow('l', label * 255)
    # cv2.waitKey(0)
