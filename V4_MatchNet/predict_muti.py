# *_*coding:utf-8 *_*
# @Author : YueMengRui
from model_mutimodel import MatchNet
import torch
from dataset_mutimodel import ImageColor, ImageScaleResize, AddGaussianNoise, ImageResize, ImagePadding, StrConverter
from torchvision import transforms
import os
import cv2
import json
import random
import numpy as np
from copy import deepcopy
import requests


def get_ocr(img_data):
    url = 'http://134.175.246.119:9350/ai/ocr/byte'

    req_data = {
        'file': img_data
    }

    try:
        resp = requests.post(url=url, files=req_data)
        return resp.json()['data']['results']
    except Exception as e:
        print(str(e))
        return ' '


def get_ocr_text(img):
    img_encode = cv2.imencode('.jpg', img)[1]

    data_encode = np.array(img_encode)

    text = get_ocr(data_encode)

    return text


class MatchDataset:

    def __init__(self, dataset_dir='/Users/yuemengrui/Data/RPAUI/web_cv_data', **kwargs):

        self.pre_processing = [ImageScaleResize(), ImageColor(), AddGaussianNoise()]

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

        return data_list

    def _get_img(self, i):
        data = self.data_list[i]

        origin_img = cv2.imread(data['img_path'])

        with open(data['box_path'], 'r', encoding='utf8') as f:
            boxes = json.load(f)

        index = random.randint(0, len(boxes) - 1)

        box = boxes[index]
        img = origin_img[box[1]:box[3], box[0]:box[2]]

        return origin_img, box, img, boxes

    def get_item(self):
        idx = random.randint(0, len(self.data_list) - 1)
        origin_img, box, small_img, boxes = self._get_img(idx)
        ori_h, ori_w = origin_img.shape[:2]

        target_x1 = max(box[0] + random.randint(-20, 6), 0)
        target_y1 = max(box[1] + random.randint(-10, 0), 0)
        target_x2 = min(box[2] + random.randint(-6, 20), ori_w)
        target_y2 = min(box[3] + random.randint(0, 10), ori_h)

        target = origin_img[target_y1:target_y2, target_x1:target_x2]

        target_text = get_ocr_text(target)

        for p in self.pre_processing:
            small_img = p(small_img)
            target = p(target)

        return origin_img, target, target_text, boxes

        # small_img = self.image_padding(small_img)
        # small_img = self.image_resize(small_img)
        # small_img = self.transform(small_img)
        #
        # target = self.image_padding(target)
        # target = self.image_resize(target)
        # target = self.transform(target)
        #
        # return small_img, target


class Predictor(object):

    def __init__(self,
                 checkpoint_path='/Users/yuemengrui/MyWork/Researches/ViT_Object_Detection/V4_MatchNet/checkpoints/mutimodel_v2.pth',
                 device='cpu'):
        self.device = torch.device(device)
        self.model = MatchNet(dim=544, depth=6, heads=8, mlp_dim=544)

        self._load_checkpoint(checkpoint_path)

        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'])

    def predict(self, img, img_text_encode, target, target_text_encode):
        img = img.to(self.device)
        img_text_encode = img_text_encode.to(self.device)
        target = target.to(self.device)
        target_text_encode = target_text_encode.to(self.device)

        with torch.no_grad():
            preds = self.model(img, img_text_encode, target, target_text_encode)
            # preds = preds.data.cpu().numpy().tolist()
            if preds.cpu().data.max(1)[1][0] == 1:
                score = float(torch.softmax(preds, dim=1)[0][1])

                return score

        return 0.0

        # return preds[0][0]


if __name__ == '__main__':
    image_padding = ImagePadding()
    image_resize = ImageResize(size=(64, 32))
    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    ])

    converter = StrConverter()

    predictor = Predictor()

    match_dataset = MatchDataset()

    origin_img, target, target_text, boxes = match_dataset.get_item()
    cv2.imshow('ori', origin_img)
    cv2.waitKey(0)

    cv2.imshow('target', target)
    cv2.waitKey(0)

    target_tensor = image_padding(target)
    target_tensor = image_resize(target_tensor)
    target_tensor = transform(target_tensor)
    target_tensor = target_tensor.unsqueeze(0)

    target_text_encode = converter.text_encode(target_text)
    target_text_encode = torch.from_numpy(target_text_encode)
    target_text_encode = target_text_encode.unsqueeze(0)

    match_boxes = []
    for box in boxes:
        # cv2.rectangle(origin_img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)
        #
        # cv2.imshow('xx', origin_img)
        # cv2.waitKey(0)
        img = origin_img[box[1]:box[3], box[0]:box[2]]
        img_text = get_ocr_text(img)
        img_text_encode = converter.text_encode(img_text)
        img_text_encode = torch.from_numpy(img_text_encode)
        img_text_encode = img_text_encode.unsqueeze(0)

        img = image_padding(img)
        img = image_resize(img)
        img = transform(img)
        img = img.unsqueeze(0)

        score = predictor.predict(img, img_text_encode, deepcopy(target_tensor), target_text_encode)
        if score > 0.9:
            match_boxes.append([box, score])

    match_boxes.sort(key=lambda x: x[1], reverse=True)

    for m_box in match_boxes:
        cv2.rectangle(origin_img, (m_box[0][0], m_box[0][1]), (m_box[0][2], m_box[0][3]), (0, 0, 255), 2)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(origin_img, str(m_box[1])[:6], (m_box[0][0], m_box[0][1]), font, 1.0, (255, 0, 0), 2)
        cv2.imshow('res', origin_img)
        cv2.waitKey(0)
