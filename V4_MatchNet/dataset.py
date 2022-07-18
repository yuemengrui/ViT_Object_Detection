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


class ImageColor(object):

    def __call__(self, img):
        if random.random() > 0.5:
            contrast = random.uniform(0.5, 1.0)
            brightness = random.randint(0, 10)
            img = cv2.addWeighted(img, contrast, img, 0, brightness)
        return img


class AddGaussianNoise(object):

    def __init__(self, mean=0, sigma=(1, 20)):
        self.mean = mean
        self.sigma = sigma

    def __call__(self, img):
        if random.random() > 0.5:
            image = np.array(img / 255, dtype=float)
            noise = np.random.normal(self.mean, random.randint(self.sigma[0], self.sigma[1]) / 255.0, image.shape)
            out = image + noise
            img = np.clip(out, 0.0, 1.0)
            img = np.uint8(img * 255.0)
        return img


class ImageScaleResize(object):

    def __init__(self, scale=0.3):
        self.scale = scale

    def __call__(self, img):
        if random.random() > 0.5:
            h, w = img.shape[:2]
            if h < w:
                min_edge = 'h'
                scale = h / w
            else:
                min_edge = 'w'
                scale = w / h

            new_scale = random.uniform(max(scale - self.scale, 0.1), min(scale + self.scale, 20))

            if random.random() > 0.5:
                new_w = w
                if min_edge == 'h':
                    new_h = int(new_w * new_scale)
                else:
                    new_h = int(new_w / new_scale)
            else:
                new_h = h
                if min_edge == 'h':
                    new_w = int(new_h / new_scale)
                else:
                    new_w = int(new_h * new_scale)

            img = cv2.resize(img, (new_w, new_h))

        return img


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

    def __init__(self, size=(64, 32)):
        self.size = size

    def __call__(self, img, **kwargs):
        # h = img.shape[0]
        # scale = self.size[1] / h
        img = cv2.resize(img, self.size)

        return img


class MatchDataset(Dataset):

    def __init__(self, dataset_dir, mode='train', **kwargs):

        self.mode = mode
        if mode == 'train':
            self.threshold = 0.7
        else:
            self.threshold = 0.2
        self.image_padding = ImagePadding()
        self.image_resize = ImageResize(size=(64, 32))

        self.pre_processing = [ImageScaleResize(), ImageColor(), AddGaussianNoise()]

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                             ])

        self.data_list = self._load_data(dataset_dir)

    def _load_data(self, dataset_dir):
        img_dir = os.path.join(dataset_dir, 'images')
        boxes_dir = os.path.join(dataset_dir, 'box_labels')

        file_list = os.listdir(img_dir)

        data_list = []
        for fil in file_list:
            file_name = fil.split('.')[0]
            img_path = os.path.join(img_dir, fil)
            box_path = os.path.join(boxes_dir, file_name + '.json')

            data_list.append({'img_path': img_path, 'box_path': box_path})

        if self.mode == 'val':
            random.shuffle(data_list)
            return data_list[:100]

        return data_list

    def _get_img(self, i):
        data = self.data_list[i]

        origin_img = cv2.imread(data['img_path'])

        with open(data['box_path'], 'r', encoding='utf8') as f:
            boxes = json.load(f)

        index = random.randint(0, len(boxes) - 1)

        box = boxes[index]
        img = origin_img[box[1]:box[3], box[0]:box[2]]

        return origin_img, box, img

    def __getitem__(self, idx):
        try:

            origin_img, box, small_img = self._get_img(idx)
            ori_h, ori_w = origin_img.shape[:2]

            if random.random() > self.threshold:
                target_x1 = max(box[0] + random.randint(-20, 6), 0)
                target_y1 = max(box[1] + random.randint(-10, 0), 0)
                target_x2 = min(box[2] + random.randint(-6, 20), ori_w)
                target_y2 = min(box[3] + random.randint(0, 10), ori_h)

                target = origin_img[target_y1:target_y2, target_x1:target_x2]

                label = torch.from_numpy(np.array([1]))
            else:
                while 1:
                    new_idx = random.randint(0, len(self.data_list) - 1)
                    if new_idx != idx:
                        break

                _, _, target = self._get_img(new_idx)
                label = torch.from_numpy(np.array([0]))

            for p in self.pre_processing:
                small_img = p(small_img)
                target = p(target)

            small_img = self.image_padding(small_img)
            small_img = self.image_resize(small_img)
            small_img = self.transform(small_img)

            target = self.image_padding(target)
            target = self.image_resize(target)
            target = self.transform(target)

            return small_img, target, label

        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(self.__len__()))

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    import torch.nn as nn

    dataset = MatchDataset(dataset_dir='/Users/yuemengrui/Data/RPAUI/train_data')
    train_loader = DataLoader(dataset, batch_size=4, shuffle=True, drop_last=True)

    criterion = nn.BCELoss()
    pred = torch.randn((4, 1)).sigmoid()
    print(pred)

    for img, target, label in train_loader:
        print(label.shape)
        print(label)
        label = label.numpy().tolist()
        print(label)

        # loss = criterion(pred, label.float())
        # print(loss)

        break
    # for i in range(10):
    #     img, target, label = dataset[i]
    # print(img.shape)
    # print(target.shape)
    # print(label)
    #
    # cv2.imshow('ori', img)
    # cv2.waitKey(0)
    #
    # cv2.imshow('target', target)
    # cv2.waitKey(0)

    # target_h, target_w = target.shape[:2]
    # print(target_h, target_w)
    #
    # x1 = int(center[0] - target_w / 2)
    # y1 = int(center[1] - target_h / 2)
    # x2 = int(center[0] + target_w / 2)
    # y2 = int(center[1] + target_h / 2)

    #     # cv2.rectangle(img, (int(center_x_min), int(center_y_min)), (int(center_x_max), int(center_y_max)), (0, 0, 255), 2)
    #     cv2.circle(img, (int(center_x), int(center_y)), 3, (255, 0, 0), 3)
    #
    #     cv2.imshow('xx', img)
    #     cv2.waitKey(0)
    #
    #     cv2.imshow('label', label * 255)
    #     cv2.waitKey(0)

    # # print(box)
    # # img_padding = ImagePadding()
    # # img_resize = ImageResize()
    # #
    # # img = cv2.imread('/Users/yuemengrui/Data/RPAUI/SAP截图及样本数据/SAP界面截图/出口运量结果界面1.PNG')
    # # print(img.shape[:2])
    # #
    # # new_img = img_padding(img)
    # #
    # # print(new_img.shape[:2])
    # #
    # # cv2.imshow('xx', new_img)
    # # cv2.waitKey(0)
    # #
    # # r_img, _ = img_resize(new_img)
    # # print(r_img.shape[:2])
    # #
    # # cv2.imshow('a', r_img)
    # # cv2.waitKey(0)
    # image_scale_resize = ImageScaleResize()
    # noise = AddGaussianNoise()
    # color = ImageColor()
    # img = cv2.imread('/Users/yuemengrui/Downloads/IMG_1621.JPG')
    # cv2.imshow('ori', img)
    # cv2.waitKey(0)
    # # print(img.shape)
    # for i in range(10):
    #     im = color(img)
    #
    #     cv2.imshow(str(i), im)
    #     cv2.waitKey(0)
