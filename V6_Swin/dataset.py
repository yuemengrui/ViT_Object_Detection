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


def crop_ok(target_box, binary, threshold=0.4):
    target_bin = binary[target_box[1]:target_box[3], target_box[0]:target_box[2]]

    t_h, t_w = target_bin.shape[:2]

    if np.sum(target_bin == 1) / (t_h * t_w) > threshold:
        return True

    return False


class ImageResize(object):

    def __init__(self, size=(832, 512)):
        self.size = size

    def __call__(self, img, **kwargs):
        h = img.shape[0]
        scale = self.size[1] / h
        img = cv2.resize(img, self.size)

        return img, scale


class ViTSegDataset(Dataset):

    def __init__(self, dataset_dir, mode='train', target_w_range=(32, 960), target_h_range=(20, 224),
                 target_w_h_rate=(0.8, 5.0), threshold=0.4, **kwargs):

        self.target_w_range = target_w_range
        self.target_h_range = target_h_range
        self.target_w_h_rate = target_w_h_rate
        self.threshold = threshold

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

        # if mode == 'val':
        #     random.shuffle(data_list)
        #     return data_list[:100]
        return data_list

    def get_crop_img(self, ori_h, ori_w, binary):
        start = time.time()
        while True:
            end = time.time()
            if end - start > 12:
                return None
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

            if target_box is None:
                return self.__getitem__(np.random.randint(self.__len__()))

            target = img[target_box[1]:target_box[3], target_box[0]:target_box[2]]

            label[target_box[1]:target_box[3], target_box[0]:target_box[2]] = 1

            img, _ = self.image_resize(img)
            img = self.transform(img)

            target, _ = self.target_resize(target)
            target = self.transform(target)

            label, _ = self.image_resize(label)
            label = torch.from_numpy(label)

            return img, target, label

        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(self.__len__()))

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    from copy import deepcopy
    import torch.nn.functional as F
    import torch.nn as nn

    dataset = ViTSegDataset(dataset_dir='/Users/yuemengrui/Data/RPAUI/train_data_new')
    # l0 = []
    # l1 = []
    num_0 = 0
    num_1 = 0
    for i in range(337):
        # s = time.time()
        img, target, label = dataset[i]

        num_1 += np.sum(label == 1)
        num_0 += np.sum(label == 0)

        # l0.append(num_0 / (512*832))
        # l1.append(num_1 / (512*832))
        # l2.append(num_1 / num_0)

    num_0 = num_0 / 337
    num_1 = num_1 / 337
    print(num_0, num_1)
    print(num_0 / num_1)
    # l.sort()
    # print(l)
    # print(np.mean(np.array(l0)))
    # print(np.mean(np.array(l1)))
    #
    # l2.sort()
    # print(l2)
    # print(np.mean(np.array(l2)))
    # im = img[500:504,500:504]
    # # cv2.imshow('xx', im)
    # # cv2.waitKey(0)
    # cv2.imwrite('./im.jpg', im)
    #
    # t = img[:2,:2]
    # cv2.imwrite('./t.jpg', t)
    # im = deepcopy(img)

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

    # cv2.rectangle(im, (target_box[0], target_box[1]), (target_box[2], target_box[3]), (0,0,255), 2)
    # cv2.imshow('xx', im)
    # cv2.waitKey(0)
    # img = cv2.imread('/Users/yuemengrui/Data/OCR/train_data/cn/cn00000006.jpg')
    # im = img[10:20, 10:20]
    # target = im[2:7, 2:7]
    # res = cv2.matchTemplate(im, target, cv2.TM_CCOEFF_NORMED)
    # print(res)
    # r = cv2.resize(res, (10, 10))
    # print(r)
    # print(res)
    # (minVal, maxVal, minLoc, maxLoc) = cv2.minMaxLoc(res)
    # (startX, startY) = maxLoc
    # endX = startX + target.shape[1]
    # endY = startY + target.shape[0]
    #
    # cv2.rectangle(im, (startX,startY), (endX,endY), (255,0,0), 2)
    # cv2.imshow('xx', im)
    # cv2.waitKey(0)
    # img = img.unsqueeze(0)
    # target = target.unsqueeze(0)
    # feature = F.conv2d(img, target)
    # print(feature)
    # bn = nn.BatchNorm2d(1)
    # f = bn(feature)
    # print(f)

    # ll.sort()
    # print(ll)
    # mean = np.mean(np.array(ll))
    # print(mean)
    # m = np.mean(np.array(l1))
    # print(m)
    # print(1 / m)
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
