# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
from model import SwinTransformer
import numpy as np
import os
import random
import cv2
import json
from torchvision import transforms


class ImageResize(object):

    def __init__(self, size=(832, 512)):
        self.size = size

    def __call__(self, img, **kwargs):
        h = img.shape[0]
        scale = self.size[1] / h
        img = cv2.resize(img, self.size)

        return img, scale


class Predictor(object):

    def __init__(self, checkpoint_path='./checkpoints/model_7744.pth', device='cpu'):
        self.device = torch.device(device)
        self.model = SwinTransformer()

        self._load_checkpoint(checkpoint_path)

        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'])

    def predict(self, img):
        img = img.to(self.device)
        with torch.no_grad():
            preds = self.model(img)
            preds = preds.data.cpu().numpy()
            pred = np.argmax(preds, axis=1)

        return pred


if __name__ == '__main__':
    image_resize = ImageResize()

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    ])

    predictor = Predictor()

    origin_img = cv2.imread('/Users/yuemengrui/Data/RPAUI/web_cv_data/images/0afb3af92ca3e735a1aa444ea1535ec9.png')

    # cv2.imshow('ori', origin_img)
    # cv2.waitKey(0)

    r_img, scale = image_resize(origin_img)
    img = transform(r_img)
    img = img.unsqueeze(0)

    pred = predictor.predict(img)
    pred = pred.squeeze(0)
    pred_img = pred.astype(np.uint8)

    cv2.imshow('pred', pred_img * 255)
    cv2.waitKey(0)

    contours, _ = cv2.findContours(pred_img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓搜索，找到
    cv2.drawContours(r_img, contours, -1, (0, 0, 255), 2)  # 绘制轮廓

    cv2.imshow("img", r_img)
    cv2.waitKey(0)

    # length = len(contours)
    # small_rects = []
    #
    # for i in range(length):
    #     cnt = contours[i]
    #
    #     # area = cv2.contourArea(cnt)
    #     # if area < 400:
    #     #     continue
    #
    #     approx = cv2.approxPolyDP(cnt, 10, True)
    #     x, y, w, h = cv2.boundingRect(approx)
    #
    #     if w < 10 or h < 10:
    #         continue
    #
    #     # w_list.append(w)
    #     # h_list.append(h)
    #
    #     small_rects.append([int(x / scale), int(y / scale), int((x + w) / scale), int((y + h) / scale)])
    #
    # small_rects.sort(key=lambda rect: rect[1])
    #
    # for b in small_rects:
    #     cv2.rectangle(origin_img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    #     cv2.imshow('x', origin_img)
    #     cv2.waitKey(0)
