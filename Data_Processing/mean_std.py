# *_*coding:utf-8 *_*
# @Author : YueMengRui
import numpy as np
import cv2
import os

img_h, img_w = 256, 256
imgs = np.zeros([img_w, img_h, 3, 1])
means, stdevs = [], []

img_dir = '/Users/yuemengrui/Data/IDP/train_dataset/images/training'
img_list = os.listdir(img_dir)
for idx, im in enumerate(img_list):
    print(idx)
    img_path = os.path.join(img_dir, im)

    img = cv2.imread(img_path)
    img = cv2.resize(img, (img_h, img_w))
    img = img[:, :, :, np.newaxis]

    imgs = np.concatenate((imgs, img), axis=3)

imgs = imgs.astype(np.float32) / 255.

for i in range(3):
    pixels = imgs[:, :, i, :].ravel()  # 拉成一行
    means.append(np.mean(pixels))
    stdevs.append(np.std(pixels))

# cv2 读取的图像格式为BGR，PIL/Skimage读取到的都是RGB不用转
means.reverse()  # BGR --> RGB
stdevs.reverse()

print("normMean = {}".format(means))
print("normStd = {}".format(stdevs))
print('transforms.Normalize(normMean = {}, normStd = {})'.format(means, stdevs))
