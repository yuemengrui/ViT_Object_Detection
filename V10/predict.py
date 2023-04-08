# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
from model import VisionTransformer
import numpy as np
import os
import random
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


class ViTSegDataset:

    def __init__(self, dataset_dir, **kwargs):
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

    def get_item(self):
        idx = random.randint(0, len(self.data_list) - 1)
        data = self.data_list[idx]

        img = cv2.imread(data['img_path'])
        boxes = data['boxes']

        target_box = boxes[random.randint(0, len(boxes) - 1)]

        target = img[target_box[1]:target_box[3], target_box[0]:target_box[2]]

        return img, target


class Predictor(object):

    def __init__(self,
                 checkpoint_path='./checkpoints/model_best.pth',
                 device='cpu'):
        self.device = torch.device(device)
        self.model = VisionTransformer()

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
    target_resize = ImageResize(size_range=(32, 64))

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.866, 0.873, 0.895), (0.278, 0.254, 0.224)),
                                    ])

    predictor = Predictor()
    dataset = ViTSegDataset('/Users/yuemengrui/Data/RPAUI/train_data_prebox')

    for _ in range(10):

        origin_img, target = dataset.get_item()

        cv2.imshow('ori', origin_img)
        cv2.waitKey(0)

        cv2.imshow('target', target)
        cv2.waitKey(0)

        img, h_scale, w_scale = image_resize(origin_img)
        img = transform(img)
        img = img.unsqueeze(0)

        target_tensor, _, _ = target_resize(target)
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

            small_rects.append([int(x / w_scale), int(y / h_scale), int((x + w) / w_scale), int((y + h) / h_scale)])

        small_rects.sort(key=lambda rect: rect[1])

        for b in small_rects:
            cv2.rectangle(origin_img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
            cv2.imshow('x', origin_img)
            cv2.waitKey(0)
