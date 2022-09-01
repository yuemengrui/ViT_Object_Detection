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

    def __call__(self, img, box):
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

        w_scale = new_w / w
        h_scale = new_h / h
        img = cv2.resize(img, (new_w, new_h))
        # label = cv2.resize(label, (new_w, new_h))
        x1, y1, x2, y2 = int(box[0] * w_scale), int(box[1] * h_scale), int(box[2] * w_scale), int(box[3] * h_scale)

        return img, [x1, y1, x2, y2]


class ImageResize(object):

    def __init__(self, size_range):
        self.size_range = size_range
        # self.padding = padding

    def __call__(self, img, box):
        h, w = img.shape[:2]
        # if h > w:
        #     new_w = h
        #     border = int((new_w - w) / 2)
        #     img = cv2.copyMakeBorder(img, 0, 0, border, border, cv2.BORDER_CONSTANT, value=self.padding)
        # else:
        #     new_h = w
        #     border = int((new_h - h) / 2)
        #     img = cv2.copyMakeBorder(img, border, border, 0, 0, cv2.BORDER_CONSTANT, value=self.padding)

        w_scale = self.size_range[1] / w
        h_scale = self.size_range[0] / h
        img = cv2.resize(img, (self.size_range[1], self.size_range[0]))
        x1, y1, x2, y2 = int(box[0] * w_scale), int(box[1] * h_scale), int(box[2] * w_scale), int(box[3] * h_scale)

        return img, [x1, y1, x2, y2]


class TargetResize(object):
    def __init__(self, size=640):
        self.size = size

    def __call__(self, img):
        h, w = img.shape[:2]

        if h > self.size or w > self.size:
            scale = self.size / max(h, w)
            new_h, new_w = int(h * scale), int(w * scale)

            img = cv2.resize(img, (new_w, new_h))

        h_border = int((self.size - h) / 2) if int((self.size - h) / 2) > 0 else 0
        w_border = int((self.size - w) / 2) if int((self.size - w) / 2) > 0 else 0
        if h_border != 0 or w_border != 0:
            img = cv2.copyMakeBorder(img, h_border, h_border, w_border, w_border, cv2.BORDER_CONSTANT, value=0)

        x1, y1, x2, y2 = w_border, h_border, w_border + w, h_border + h
        nh, nw = img.shape[:2]
        if nh != self.size or nw != self.size:
            h_scale = self.size / nh
            w_scale = self.size / nw
            img = cv2.resize(img, (self.size, self.size))
            x1, y1, x2, y2 = int(w_border * w_scale), int(h_border * h_scale), int((w_border + w) * w_scale), int(
                (h_border + h) * h_scale)

        return img, [x1, y1, x2, y2]


class ViTSegDataset(Dataset):

    def __init__(self, dataset_dir, mode='train', **kwargs):

        self.image_resize = ImageResize(size_range=(640, 640))
        self.target_resize = TargetResize(size=640)
        self.image_random_scale_resize = ImageRandomScaleResize()

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.866, 0.873, 0.895), (0.278, 0.254, 0.224)),
                                             ])

        self.data_list = self._load_data(dataset_dir, mode)

    def _load_data(self, dataset_dir, mode='train'):
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

        if mode == 'val':
            return data_list[:100]

        return data_list

    def __getitem__(self, idx):
        try:
            data = self.data_list[idx]

            img = cv2.imread(data['img_path'])
            boxes = data['boxes']

            # ori_h, ori_w = img.shape[:2]

            # label = np.uint8(np.zeros((ori_h, ori_w)))

            target_box = boxes[random.randint(0, len(boxes) - 1)]

            target = img[target_box[1]:target_box[3], target_box[0]:target_box[2]]

            # label[target_box[1]:target_box[3], target_box[0]:target_box[2]] = 1

            img, label_box = self.image_random_scale_resize(img, target_box)

            img, label_box = self.image_resize(img, label_box)

            target, box = self.target_resize(target)

            # label = self.image_resize(label)

            img = self.transform(img)
            target = self.transform(target)
            # label = torch.from_numpy(label)

            return img, target, torch.from_numpy(np.array(box)), torch.from_numpy(np.array(label_box))

        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(self.__len__()))

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    dataset = ViTSegDataset(dataset_dir='/Users/yuemengrui/Data/RPAUI/train_data_prebox')
    # for _ in range(10):
    # img, target, box, label_box = dataset[100]

    # num_0 = 0
    # num_1 = 0
    # for i in range(367):
    #     print(i)
    #     #     # s = time.time()
    #     img, target, label = dataset[i]
    #
    #     num_1 += np.sum(label == 1)
    #     num_0 += np.sum(label == 0)
    #
    # num_0 = num_0 / 367
    # num_1 = num_1 / 367
    # print(num_0, num_1, num_0 / num_1)
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
    # print(img.shape)
    # print(target.shape)
    # cv2.imshow('xx', img)
    # cv2.waitKey(0)
    #
    # cv2.rectangle(img, (label_box[0], label_box[1]), (label_box[2], label_box[3]), (0, 0, 255), 2)
    # cv2.imshow('x', img)
    # cv2.waitKey(0)
    #
    # cv2.imshow('t', target)
    # cv2.waitKey(0)
    #
    # cv2.rectangle(target, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
    # cv2.imshow('tt', target)
    # cv2.waitKey(0)
    #
    # cv2.imshow('l', label * 255)
    # cv2.waitKey(0)

    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

    for (img, target, box, label_box) in train_loader:
        print(img.shape)
        print(target.shape)
        print(box, type(box))
        print(label_box, type(label_box))
        print(box[0], type(box[0]))
        print(box[0][0], type(box[0][0]))
        print(box[0][2] - box[0][0])
        print(box[0][3] - box[0][1])

        break
