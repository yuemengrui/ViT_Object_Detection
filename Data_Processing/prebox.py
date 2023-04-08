# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import numpy as np
import cv2
import random
import time
import json


def crop_ok(target_box, binary, threshold=0.4):
    target_bin = binary[target_box[1]:target_box[3], target_box[0]:target_box[2]]

    t_h, t_w = target_bin.shape[:2]

    if np.sum(target_bin == 1) / (t_h * t_w) > threshold:
        return True

    return False


def load_data(dataset_dir):
    img_dir = os.path.join(dataset_dir, 'images')
    labels_dir = os.path.join(dataset_dir, 'labels')
    prebox_dir = os.path.join(dataset_dir, 'prebox_labels')

    file_list = os.listdir(img_dir)

    data_list = []
    for fil in file_list:
        file_name = fil.split('.')[0]
        img_path = os.path.join(img_dir, fil)
        label_path = os.path.join(labels_dir, file_name + '.png')
        prebox_path = os.path.join(prebox_dir, file_name + '.json')

        data_list.append(
            {'img_path': img_path, 'label_path': label_path, 'prebox_path': prebox_path, 'file_name': file_name})

    return data_list


def get_crop_img(ori_h, ori_w, binary, target_w_range=(32, 960), target_h_range=(22, 224), target_edge_max_scale=4.36,
                 threshold=0.3):
    # start = time.time()
    while True:
        # end = time.time()
        # if end - start > 12:
        #     return None
        x1 = random.randint(0, ori_w - target_w_range[0])
        y1 = random.randint(0, ori_h - target_h_range[0])
        target_w = random.randint(target_w_range[0], target_w_range[1])
        target_h = random.randint(target_h_range[0], target_h_range[1])

        x2 = min(ori_w, x1 + target_w)
        y2 = min(ori_h, y1 + target_h)

        w = x2 - x1
        h = y2 - y1

        if w / h < 0.8:
            continue

        scale = max(h, w) / min(h, w)
        if scale <= target_edge_max_scale:
            if crop_ok([x1, y1, x2, y2], binary, threshold):
                return [x1, y1, x2, y2]


if __name__ == '__main__':
    data_list = load_data('/Users/yuemengrui/Data/RPAUI/train_data_prebox')

    for i in range(len(data_list)):
        print(i)
        data = data_list[i]
        img = cv2.imread(data['img_path'])
        binary = cv2.imread(data['label_path'], -1)
        ori_h, ori_w = img.shape[:2]

        boxes = []
        start = time.time()
        for _ in range(2000):
            if time.time() - start > 200:
                break

            target_box = get_crop_img(ori_h, ori_w, binary)
            # target = img[target_box[1]:target_box[3], target_box[0]:target_box[2]]
            if target_box not in boxes:
                boxes.append(target_box)

        print(data['file_name'], len(boxes), time.time() - start)

        with open(data['prebox_path'], 'w') as f:
            json.dump(boxes, f)
