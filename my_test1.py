# *_*coding:utf-8 *_*
# @Author : YueMengRui
import cv2
import numpy as np
import os
import json


def create_boxes(img_path):
    label_img = cv2.imread(img_path, -1)
    horizontalStructure2 = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    col_erosion = cv2.erode(label_img, horizontalStructure2, iterations=1)
    col_dilation = cv2.dilate(col_erosion, horizontalStructure2, iterations=2)
    contours, _ = cv2.findContours(col_dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # 轮廓搜索，找到
    # cv2.drawContours(ori_img, contours, -1, (0, 0, 255), 2)  # 绘制轮廓
    #
    # cv2.imshow("img", ori_img)
    # cv2.waitKey(0)

    length = len(contours)
    small_rects = []
    w_list = []
    h_list = []
    for i in range(length):
        cnt = contours[i]

        area = cv2.contourArea(cnt)
        if area < 400:
            continue

        approx = cv2.approxPolyDP(cnt, 10, True)
        x, y, w, h = cv2.boundingRect(approx)

        if w < 20 or h < 15:
            continue

        w_list.append(w)
        h_list.append(h)

        small_rects.append([x, y, x + w, y + h])

    small_rects.sort(key=lambda rect: rect[1])
    print(len(small_rects))
    w_mean = np.mean(np.array(w_list))
    h_mean = np.mean(np.array(h_list))
    w_min = min(w_list)
    w_max = max(w_list)
    h_min = min(h_list)
    h_max = max(h_list)

    print('w_min:{} w_max:{} w_mean:{}'.format(w_min, w_max, w_mean))
    print('h_min:{} h_max:{} h_mean:{}'.format(h_min, h_max, h_mean))

    # for b in small_rects:
    #     cv2.rectangle(ori_img, (b[0], b[1]), (b[2], b[3]), (0, 0, 255), 2)
    #     cv2.imshow('xx', ori_img)
    #     cv2.waitKey(0)

    return small_rects, w_min, w_max, w_mean, h_min, h_max, h_mean


if __name__ == '__main__':
    # ori_img = cv2.imread('/Users/yuemengrui/Data/RPAUI/web_cv_data/images/0afb3af92ca3e735a1aa444ea1535ec9.png')
    # create_boxes(ori_img, '/Users/yuemengrui/Data/RPAUI/web_cv_data/labels/0afb3af92ca3e735a1aa444ea1535ec9.png')

    # data_dir = '/Users/yuemengrui/Data/RPAUI/web_cv_data/labels'
    #
    # box_dir = '/Users/yuemengrui/Data/RPAUI/web_cv_data/box_labels'
    #
    # label_list = os.listdir(data_dir)
    #
    # min_w = 9999
    # min_h = 9999
    # max_w = 0
    # max_h = 0
    # w_mean_list = []
    # h_mean_list = []
    #
    # for l in label_list:
    #     file_name = l.split('.')[0]
    #
    #     img_path = os.path.join(data_dir, l)
    #     print('----------------------------------------')
    #     print(img_path)
    #     boxes, w_min, w_max, w_mean, h_min, h_max, h_mean = create_boxes(img_path)
    #     print('----------------------------------------')
    #
    #     w_mean_list.append(w_mean)
    #     h_mean_list.append(h_mean)
    #
    #     if w_min < min_w:
    #         min_w = w_min
    #
    #     if h_min < min_h:
    #         min_h = h_min
    #
    #     if w_max > max_w:
    #         max_w = w_max
    #
    #     if h_max > max_h:
    #         max_h = h_max
    #
    #     box_path = os.path.join(box_dir, file_name + '.json')
    #     with open(box_path, 'w', encoding='utf8') as f:
    #         json.dump(boxes, f)
    #
    # mean_w = np.mean(np.array(w_mean_list))
    # mean_h = np.mean(np.array(h_mean_list))
    #
    # print('w: ', min_w, max_w, mean_w)  # 1 1804 61.48
    # print('h: ', min_h, max_h, mean_h)  # 1 542 19.60

    img_dir = '/Users/yuemengrui/Data/RPAUI/train_data/images'
    box_dir = '/Users/yuemengrui/Data/RPAUI/train_data/box_labels'

    img_list = os.listdir(img_dir)

    for i in img_list:
        file_name = i.split('.')[0]

        img_path = os.path.join(img_dir, i)
        box_path = os.path.join(box_dir, file_name + '.json')

        with open(box_path, 'r') as f:
            box_list = json.load(f)

        print(file_name)

        img = cv2.imread(img_path)

        for box in box_list:
            w = box[2] - box[0]
            h = box[3] - box[1]
            print('w: {} h:{} w/h:{}'.format(w, h, w/h))

            cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), 2)

            cv2.imshow('xx', img)
            cv2.waitKey(0)
