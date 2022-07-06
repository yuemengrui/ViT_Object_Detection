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


def crop_ok(target_box, boxes, threshold=0.5):
    for box in boxes:
        if iou(target_box, box) > threshold:
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


class ViTDetDataset(Dataset):

    def __init__(self, dataset_dir, mode='train', target_size_range=(10, 300), threshold=0.4, **kwargs):

        self.target_size_range = target_size_range
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
        boxes_dir = os.path.join(dataset_dir, 'box_labels')

        file_list = os.listdir(img_dir)

        data_list = []
        for fil in file_list:
            file_name = fil.split('.')[0]
            img_path = os.path.join(img_dir, fil)
            box_path = os.path.join(boxes_dir, file_name + '.json')

            data_list.append({'img_path': img_path, 'box_path': box_path})

        if mode == 'val':
            random.shuffle(data_list)
            return data_list[:50]
        return data_list

    def get_crop_img(self, ori_h, ori_w, boxes):
        while True:
            x1 = random.randint(0, ori_w - self.target_size_range[0])
            y1 = random.randint(0, ori_h - self.target_size_range[0])
            target_w = random.randint(self.target_size_range[0], self.target_size_range[1])
            target_h = random.randint(self.target_size_range[0], self.target_size_range[1])

            x2 = min(ori_w, x1 + target_w)
            y2 = min(ori_h, y1 + target_h)

            if crop_ok([x1, y1, x2, y2], boxes, self.threshold):
                return [x1, y1, x2, y2]

    def __getitem__(self, idx):
        try:
            data = self.data_list[idx]

            img = cv2.imread(data['img_path'])
            with open(data['box_path'], 'r', encoding='utf8') as f:
                boxes = json.load(f)

            ori_h, ori_w = img.shape[:2]
            target_box = self.get_crop_img(ori_h, ori_w, boxes)

            center_x = int(target_box[0] + (target_box[2] - target_box[0]) / 2)
            center_y = int(target_box[1] + (target_box[3] - target_box[1]) / 2)

            center_x_min = int(target_box[0] + (target_box[2] - target_box[0]) / 4)
            center_x_max = int(target_box[0] + 3 * (target_box[2] - target_box[0]) / 4)
            center_y_min = int(target_box[1] + (target_box[3] - target_box[1]) / 4)
            center_y_max = int(target_box[1] + 3 * (target_box[3] - target_box[1]) / 4)

            target = img[target_box[1]:target_box[3], target_box[0]:target_box[2]]

            img = self.image_padding(img)
            img, scale = self.image_resize(img)
            h, w = img.shape[:2]
            c_x = center_x * scale / w
            c_y = center_y * scale / h

            c_x_min = center_x_min * scale / w
            c_y_min = center_y_min * scale / h
            c_x_max = center_x_max * scale / w
            c_y_max = center_y_max * scale / h

            img = self.transform(img)

            target = self.image_padding(target)
            target, _ = self.target_resize(target)
            target = self.transform(target)

            label = torch.from_numpy(np.array([c_x, c_y, 1]))

            return img, target, label, torch.from_numpy(np.array([c_x_min, c_x_max, c_y_min, c_y_max]))

        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(self.__len__()))

    def __len__(self):
        return len(self.data_list)

if __name__ == '__main__':
    dataset = ViTDetDataset(dataset_dir='/Users/yuemengrui/Data/RPAUI/web_cv_data')
    train_loader = DataLoader(dataset, batch_size=2, shuffle=True, drop_last=True)

    for img, target, label, center_range in train_loader:
        print(img.shape)
        print(target.shape)
        print(label.shape)
        print(label)
        print(center_range.numpy().tolist())
        center_range = center_range.numpy().tolist()
        preds = [[0.508238673210144, 0.4653865396976471, 0.49629437923431396], [0.2,0.2,0.2]]

        for i in range(len(preds)):
            print("==========================")
            print(preds[i])
            print(center_range[i])

            pred_c_x = preds[i][0]
            pred_c_y = preds[i][1]

            print(pred_c_x, pred_c_y)

            label_c_x_min = center_range[i][0]
            label_c_x_max = center_range[i][1]
            label_c_y_min = center_range[i][2]
            label_c_y_max = center_range[i][3]

            print(label_c_x_min, label_c_x_max)
            print(label_c_y_min, label_c_y_max)


        # x_min = center_range[0][0]
        # x_max = center_range[0][1]
        # y_min = center_range[1][0]
        # y_max = center_range[1][1]
        # print(x_min, x_max)
        # print(y_min, y_max)
        # print(x_min < 0.3 < x_max)
        # print(x_min < 0.4 < x_max)
        # print(x_min < 0.5 < x_max)
        # print(x_min < 0.6 < x_max)
        break

# img, target, center_x, center_y, center_x_min, center_y_min, center_x_max, center_y_max = dataset[0]
# print(img.shape)
# print(target.shape)
#
# print(center_x, center_y)
# print(center_x_min, center_x_max)
# print(center_y_min, center_y_max)
#
# # cv2.imshow('ori', img)
# # cv2.waitKey(0)
#
# cv2.imshow('target', target)
# cv2.waitKey(0)
#
# # target_h, target_w = target.shape[:2]
# # print(target_h, target_w)
# #
# # x1 = int(center[0] - target_w / 2)
# # y1 = int(center[1] - target_h / 2)
# # x2 = int(center[0] + target_w / 2)
# # y2 = int(center[1] + target_h / 2)
#
# cv2.rectangle(img, (int(center_x_min), int(center_y_min)), (int(center_x_max), int(center_y_max)), (0, 0, 255), 2)
# cv2.circle(img, (center_x, center_y), 3, (255, 0, 0), 3)
#
# cv2.imshow('xx', img)
# cv2.waitKey(0)
#
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
