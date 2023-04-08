# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
from model_2 import ViT
import numpy as np
import os
import random
import cv2
import json
from torchvision import transforms


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
        if box[0] > target_box[2]:
            continue

        if box[2] < target_box[0]:
            continue

        if box[1] > target_box[3]:
            break

        if iou(target_box, box) > threshold:
            return True

    return False


class ViTSegDataset:

    def __init__(self, dataset_dir='/Users/yuemengrui/Data/RPAUI/train_data', mode='train', target_w_range=(30, 1000),
                 target_h_range=(20, 350),
                 target_w_h_rate=(0.7, 20), threshold=0.6, **kwargs):

        self.target_w_range = target_w_range
        self.target_h_range = target_h_range
        self.target_w_h_rate = target_w_h_rate
        self.threshold = threshold

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
                if crop_ok([x1, y1, x2, y2], boxes, self.threshold):
                    return [x1, y1, x2, y2]

    def get_item(self):
        idx = random.randint(0, len(self.data_list) - 1)
        data = self.data_list[idx]

        img = cv2.imread(data['img_path'])
        with open(data['box_path'], 'r', encoding='utf8') as f:
            boxes = json.load(f)

        ori_h, ori_w = img.shape[:2]

        target_box = self.get_crop_img(ori_h, ori_w, boxes)

        target = img[target_box[1]:target_box[3], target_box[0]:target_box[2]]

        # img = self.image_padding(img)
        # img, _ = self.image_resize(img)
        # img = self.transform(img)
        #
        # target = self.image_padding(target)
        # target, _ = self.target_resize(target)
        # target = self.transform(target)

        return img, target


class Predictor(object):

    def __init__(self,
                 checkpoint_path='/Users/yuemengrui/MyWork/Researches/ViT_Object_Detection/V3/checkpoints/model_best_0.5239.pth',
                 device='cpu'):
        self.device = torch.device(device)
        self.model = ViT(dim=512, depth=6, heads=8, mlp_dim=512)

        self._load_checkpoint(checkpoint_path)

        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'])

    def predict(self, img, target):
        img = img.to(self.device)
        target = target.to(self.device)
        with torch.no_grad():
            preds = self.model(img, target)
            preds = preds.data.cpu().numpy()
            pred = np.argmax(preds, axis=1)

        return pred


if __name__ == '__main__':
    image_padding = ImagePadding()
    image_resize = ImageResize()
    target_resize = ImageResize(size=(64, 32))

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    ])

    ViTdataset = ViTSegDataset()

    predictor = Predictor()

    origin_img, target = ViTdataset.get_item()

    cv2.imshow('ori', origin_img)
    cv2.waitKey(0)

    cv2.imshow('target', target)
    cv2.waitKey(0)

    img = image_padding(origin_img)
    img, scale = image_resize(img)
    img = transform(img)
    img = img.unsqueeze(0)

    target_tensor = image_padding(target)
    target_tensor, _ = target_resize(target_tensor)
    target_tensor = transform(target_tensor)
    target_tensor = target_tensor.unsqueeze(0)

    pred = predictor.predict(img, target_tensor)
    pred = pred.squeeze(0)
    pred_img = pred.astype(np.uint8)

    cv2.imshow('pred', pred_img * 255)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(pred_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓搜索，找到
    # cv2.drawContours(pred_img, contours, -1, (0, 0, 255), 2)  # 绘制轮廓
    #
    # cv2.imshow("img", pred_img)
    # cv2.waitKey(0)

    length = len(contours)
    small_rects = []

    for i in range(length):
        cnt = contours[i]

        # area = cv2.contourArea(cnt)
        # if area < 400:
        #     continue

        approx = cv2.approxPolyDP(cnt, 10, True)
        x, y, w, h = cv2.boundingRect(approx)

        if w < 10 or h < 10:
            continue

        # w_list.append(w)
        # h_list.append(h)

        small_rects.append([int(x / scale), int(y / scale), int((x + w) / scale), int((y + h) / scale)])

    small_rects.sort(key=lambda rect: rect[1])

    for b in small_rects:
        cv2.rectangle(origin_img, (b[0],b[1]), (b[2], b[3]), (0,0,255), 2)
        cv2.imshow('x', origin_img)
        cv2.waitKey(0)
