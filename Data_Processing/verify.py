# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import cv2


image_dir = '/Users/yuemengrui/Data/RPAUI/train_data_new/images'
label_dir = '/Users/yuemengrui/Data/RPAUI/train_data_new/labels'

img_list = os.listdir(image_dir)

for i in img_list:
    file_name = i.split('.')[0]

    img_path = os.path.join(image_dir, i)
    label_path = os.path.join(label_dir, file_name + '.png')

    img = cv2.imread(img_path)
    label = cv2.imread(label_path)

    res = cv2.addWeighted(img, 0.4, label*255, 0.6, 0)

    cv2.imshow('xx', res)
    cv2.waitKey(0)
