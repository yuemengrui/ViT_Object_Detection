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


class ImageResize(object):

    def __init__(self, size_range):
        self.size_range = size_range

    def __call__(self, img, **kwargs):
        img = cv2.resize(img, (self.size_range[1], self.size_range[0]))

        return img


class ViTSegDataset(Dataset):

    def __init__(self, dataset_dir, mode='train', **kwargs):

        self.image_resize = ImageResize(size_range=(512, 1024))
        self.target_resize = ImageResize(size_range=(32, 64))

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

            img = self.image_resize(img)

            target = self.target_resize(target)

            label = self.image_resize(label)

            img = self.transform(img)
            target = self.transform(target)
            label = torch.from_numpy(label)

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
    print(num_0, num_1, num_0 / num_1)  # 86.496 86.731 94.639 96.586 88.786
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
