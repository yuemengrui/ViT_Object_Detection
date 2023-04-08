# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
from model import ViT
import numpy as np
import time
import random
import os
import cv2
import json
from torchvision import transforms


class ImageResize(object):

    def __init__(self, size_range):
        self.size_range = size_range

    def __call__(self, img, **kwargs):
        h, w = img.shape[:2]
        h_scale = self.size_range[0] / h
        w_scale = self.size_range[1] / w
        img = cv2.resize(img, (self.size_range[1], self.size_range[0]))

        return img, h_scale, w_scale


def crop_ok(target_box, binary, threshold=0.4):
    target_bin = binary[target_box[1]:target_box[3], target_box[0]:target_box[2]]

    t_h, t_w = target_bin.shape[:2]

    if np.sum(target_bin == 1) / (t_h * t_w) > threshold:
        return True

    return False


def get_crop_img(ori_h, ori_w, binary, h_range, w_range):
    start = time.time()
    while True:
        end = time.time()
        if end - start > 12:
            return None
        x1 = random.randint(0, ori_w - w_range[0])
        y1 = random.randint(0, ori_h - h_range[0])
        target_w = random.randint(w_range[0], w_range[1])
        target_h = random.randint(h_range[0], h_range[1])

        x2 = min(ori_w, x1 + target_w)
        y2 = min(ori_h, y1 + target_h)

        w = x2 - x1
        h = y2 - y1
        rate = w / h
        if 0.8 <= rate <= 5.0:
            if crop_ok([x1, y1, x2, y2], binary, 0.3):
                return [x1, y1, x2, y2]


def create_target(targets, origin_img, ori_h, ori_w, binary, target_h_range, target_w_range,
                  origin_point_list):
    # target_num = random.randint(1, 16)
    target_num = 1
    box_list = []
    for n in range(target_num):

        target_box = get_crop_img(ori_h, ori_w, binary, target_h_range, target_w_range)

        if target_box is not None:
            target = origin_img[target_box[1]:target_box[3], target_box[0]:target_box[2]]
            # cv2.imshow('t', target)
            # cv2.waitKey(0)
            t_h, t_w = target.shape[:2]
            origin_p = origin_point_list[n]
            x1 = origin_p[0]
            y1 = origin_p[1]
            x2 = x1 + t_w
            y2 = y1 + t_h
            targets[y1:y2, x1:x2] = target
            # cv2.imshow('ts', targets)
            # cv2.waitKey(0)
            box_list.append(target_box)

    return targets, box_list


class Predictor(object):

    def __init__(self,
                 checkpoint_path='./checkpoints/model_best_3530.pth',
                 device='cpu'):
        self.device = torch.device(device)
        self.model = ViT()

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
    image_resize = ImageResize(size_range=(512, 832))

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.866, 0.873, 0.895), (0.278, 0.254, 0.224)),
                                    ])

    predictor = Predictor()
    data_dir = '/Users/yuemengrui/Data/RPAUI/train_data_new'
    img_name = '8db7933e7d14c07b73687bd1a589269d.png'
    img_path = os.path.join(data_dir, 'images', img_name)
    label_path = os.path.join(data_dir, 'labels', img_name.split('.')[0] + '.png')

    origin_img = cv2.imread(img_path)
    binary = cv2.imread(label_path, -1)
    ori_h, ori_w = origin_img.shape[:2]
    target_w_max = int(ori_w / 4)
    target_h_max = int(ori_h / 4)

    origin_point_list = []
    for i in range(16):
        row = i // 4
        col = i % 4
        x = int(ori_w / 4 * col)
        y = int(ori_h / 4 * row)
        origin_point_list.append((x, y))

    target_w_range = (32, target_w_max)
    target_h_range = (20, target_h_max)

    target = np.zeros_like(origin_img)

    target, box_list = create_target(target, origin_img, ori_h, ori_w, binary, target_h_range, target_w_range,
                                     origin_point_list)

    cv2.imshow('img', origin_img)
    cv2.waitKey(0)

    cv2.imshow('target', target)
    cv2.waitKey(0)

    im, h_scale, w_scale = image_resize(origin_img)
    im = transform(im)
    im = im.unsqueeze(0)

    target_tensor, _, _ = image_resize(target)
    target_tensor = transform(target_tensor)
    target_tensor = target_tensor.unsqueeze(0)

    pred = predictor.predict(im, target_tensor)
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

        small_rects.append([int(x / w_scale), int(y / h_scale), int((x + w) / w_scale), int((y + h) / h_scale)])

    small_rects.sort(key=lambda rect: rect[1])

    for b in small_rects:
        cv2.rectangle(origin_img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
        cv2.imshow('x', origin_img)
        cv2.waitKey(0)
