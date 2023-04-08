# *_*coding:utf-8 *_*
# @Author : YueMengRui
import numpy as np
import cv2
import os
import json
import shutil

data_dir = '/Users/yuemengrui/Data/RPAUI/img_data'
new_img_dir = '/Users/yuemengrui/Data/RPAUI/train_data_img/images'
new_label_dir = '/Users/yuemengrui/Data/RPAUI/train_data_img/labels'

data_list = os.listdir(data_dir)

for i in data_list:
    if i.endswith('.json'):
        file_name = i.split('.')[0]
        img_path = os.path.join(data_dir, file_name + '.png')
        img = cv2.imread(img_path)

        h, w = img.shape[:2]

        label = np.uint8(np.zeros((h, w)))

        with open(os.path.join(data_dir, i), 'r') as f:
            label_data = json.load(f)

        for j in label_data['shapes']:
            x1 = int(min(j['points'][0][0], j['points'][1][0]))
            x2 = int(max(j['points'][0][0], j['points'][1][0]))
            y1 = int(min(j['points'][0][1], j['points'][1][1]))
            y2 = int(max(j['points'][0][1], j['points'][1][1]))
            label[y1:y2, x1:x2] = 1

        area = np.sum(label == 1)
        a = np.sum(label == 0)

        if area / a > 0.1:
            new_img_path = os.path.join(new_img_dir, file_name + '.png')
            new_label_path = os.path.join(new_label_dir, file_name + '.png')
            shutil.copy(img_path, new_img_path)

            cv2.imwrite(new_label_path, label)
