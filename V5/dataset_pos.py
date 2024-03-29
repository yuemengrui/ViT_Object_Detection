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


def crop_ok(target_box, binary, threshold=0.6):
    target_bin = binary[target_box[1]:target_box[3], target_box[0]:target_box[2]]

    t_h, t_w = target_bin.shape[:2]

    if np.sum(target_bin == 1) / (t_h * t_w) > threshold:
        return True

    return False


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
            border = int((new_w - ori_w) / 2)
            img = cv2.copyMakeBorder(img, 0, 0, border, border, cv2.BORDER_CONSTANT, value=(0, 0, 0))
        elif ori_h / ori_w < 0.5:
            new_h = int(ori_w / 2)
            border = int((new_h - ori_h) / 2)
            img = cv2.copyMakeBorder(img, border, border, 0, 0, cv2.BORDER_CONSTANT, value=(0, 0, 0))

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

    def __init__(self, dataset_dir, mode='train', target_w_range=(32, 960), target_h_range=(20, 224),
                 target_w_h_rate=(0.7, 20), threshold=0.7, **kwargs):

        self.target_w_range = target_w_range
        self.target_h_range = target_h_range
        self.target_w_h_rate = target_w_h_rate
        self.threshold = threshold

        self.image_random_scale_resize = ImageRandomScaleResize()
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
            return data_list[:100]
        return data_list

    def get_crop_img(self, ori_h, ori_w, binary):
        # start = time.time()
        while True:
            # end = time.time()
            # if end - start > 10:
            #     return None
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

            c_x = (target_box[0] + (target_box[2] - target_box[0]) / 2) / ori_w
            c_y = (target_box[1] + (target_box[3] - target_box[1]) / 2) / ori_h

            target = img[target_box[1]:target_box[3], target_box[0]:target_box[2]]

            label[target_box[1]:target_box[3], target_box[0]:target_box[2]] = 1

            img, label = self.image_random_scale_resize(img, label)

            img = self.image_padding(img)
            img, _ = self.image_resize(img)
            img = self.transform(img)

            target = self.image_padding(target)
            target, _ = self.target_resize(target)
            target = self.transform(target)

            label = self.image_padding(label)
            label, _ = self.image_resize(label)
            label = torch.from_numpy(label)

            return img, target, label, torch.from_numpy(np.array([c_x, c_y], dtype=np.float32))

        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(self.__len__()))

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    dataset = ViTSegDataset(dataset_dir='/Users/yuemengrui/Data/RPAUI/train_data')
    # for i in range(20):
    # img, target, label, pos = dataset[1]
#     print(img.shape)
#     print(label.shape)
#
#     cv2.imshow('xx', img)
#     cv2.waitKey(0)
#
#     cv2.imshow('t', target)
#     cv2.waitKey(0)
#
#     cv2.imshow('l', label * 255)
#     cv2.waitKey(0)
#
#     label = label * 255
#     label = cv2.merge((label, label, label))
#     im = cv2.addWeighted(img, 0.4, label, 0.6, 0)
#     cv2.imshow('xxx', im)
#     cv2.waitKey(0)
# #     rate = 0
#     start = time.time()
#     for i in range(100):
#         img, target, label = dataset[i]
#     print(time.time()-start)
#     #     dataset = ViTDetDataset(dataset_dir='/Users/yuemengrui/Data/RPAUI/web_cv_data')
    train_loader = DataLoader(dataset, batch_size=64, shuffle=True, drop_last=True)
#
    for img, target, label, pos in train_loader:
        print(img.shape)
        print(target.shape)
        print(label.shape)
        print(pos.shape)
        break
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


# w/h_scale: 1.25 1.33 1.37 1.6  1.667 1.778 2.13
# 0.5 0.75 1.0 1.25  1.5  1.75  2.0  4.0
