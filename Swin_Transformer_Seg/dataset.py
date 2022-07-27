# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import random
import time
from torchvision import transforms
from torch.utils.data import DataLoader


class RandomCrop(object):

    def __init__(self, size=(512, 832)):
        self.size = size

    def __call__(self, img, label, **kwargs):
        h, w = img.shape[:2]
        h_range = h - self.size[0]
        w_range = w - self.size[1]

        x1 = random.randint(0, w_range)
        y1 = random.randint(0, h_range)

        x2 = x1 + self.size[1]
        y2 = y1 + self.size[0]

        img = img[y1:y2, x1:x2]

        label = label[y1:y2, x1:x2]

        return img, label


class ImageResize(object):

    def __init__(self, size=(832, 512)):
        self.size = size

    def __call__(self, img, label, **kwargs):
        # h = img.shape[0]
        # scale = self.size[1] / h
        img = cv2.resize(img, self.size)
        label = cv2.resize(label, self.size)

        return img, label


class ViTSegDataset(Dataset):

    def __init__(self, dataset_dir, mode='train', **kwargs):

        self.image_resize = ImageResize()
        self.image_crop = RandomCrop()

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
            return data_list[:100]
        return data_list

    def __getitem__(self, idx):
        try:
            data = self.data_list[idx]

            img = cv2.imread(data['img_path'])
            label = cv2.imread(data['label_path'], -1)

            if random.random() > 0.5:
                img, label = self.image_crop(img, label)

            img, label = self.image_resize(img, label)

            img = self.transform(img)
            label = torch.from_numpy(label)

            return img, label

        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(self.__len__()))

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    dataset = ViTSegDataset(dataset_dir='/Users/yuemengrui/Data/RPAUI/web_cv_data')

    for i in range(10):
        #     s = time.time()
        img, label = dataset[i]
        print(img.shape)
        print(label.shape)

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

    # print(1/mean)
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
#     image_padding = ImagePadding()
#
#     img = cv2.imread('/Users/yuemengrui/Data/RPAUI/train_data/images/a9e406c183a1eba6e6de84680c6b50ff.png')
#
#     print(img.shape)
#
#     im = image_padding(img)
#     print(im.shape)
#
#     cv2.imshow('xx', im)
#     cv2.waitKey(0)
