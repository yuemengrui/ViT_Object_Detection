# *_*coding:utf-8 *_*
# @Author : YueMengRui
import cv2
import numpy as np
import os
import shutil


def iou(box1, box2):
    """
    两个框（二维）的 iou 计算
    注意：边框以左上为原点
    box:[x1,y1,x2,y2],依次为左上右下坐标
    """

    w = max(0, min(box1[2], box2[2]) - max(box1[0], box2[0]))
    h = max(0, min(box1[3], box2[3]) - max(box1[1], box2[1]))
    area_box1 = ((box1[2] - box1[0]) * (box1[3] - box1[1]))
    area_box2 = ((box2[2] - box2[0]) * (box2[3] - box2[1]))
    inter = w * h
    union = area_box1 + area_box2 - inter
    iou = inter / union
    return iou


def NMS(boxes, thresh=0.3):
    true_boxes = []

    while len(boxes) > 0:
        true_boxes.append(boxes[0])
        boxes.pop(0)
        if len(boxes) == 0:
            break

        for box in boxes[::-1]:
            if box[0] > true_boxes[-1][0] and box[1] > true_boxes[-1][1] and box[2] < true_boxes[-1][2] and box[3] < \
                    true_boxes[-1][3]:
                boxes.remove(box)
                continue
            if box[0] > true_boxes[-1][2]:
                continue

            if box[2] < true_boxes[-1][0]:
                continue

            if box[1] > true_boxes[-1][3]:
                continue

            if box[3] < true_boxes[-1][1]:
                continue

            if iou(box, true_boxes[-1]) > thresh:
                boxes.remove(box)

    return true_boxes


def preprocess(gray):
    # 1. Sobel算子，x方向求梯度
    # sobel = cv2.Sobel(gray, cv2.CV_8U, 1, 0, ksize=3)
    # blur_img = cv2.GaussianBlur(gray, (1, 3), 0)
    # 2. 二值化
    ret, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_OTSU + cv2.THRESH_BINARY)
    # 3. 膨胀和腐蚀操作的核函数
    # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 2))
    kernel2 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # 4. 腐蚀一次
    erosion = cv2.erode(binary, kernel2, iterations=1)

    # 5. 膨胀一次，让轮廓突出
    dilation = cv2.dilate(erosion, kernel2, iterations=1)
    return dilation


def findTextRegion(dilation):
    region = []

    # 1. 查找轮廓
    contours, hierarchy = cv2.findContours(dilation, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    # 2. 筛选那些面积小的
    for i in range(len(contours)):
        cnt = contours[i]
        # 计算该轮廓的面积
        area = cv2.contourArea(cnt)

        # 面积小的都筛选掉
        if area < 200:
            continue

        # 找到最小的矩形，该矩形可能有方向
        rect = cv2.minAreaRect(cnt)

        # box是四个点的坐标
        box = cv2.boxPoints(rect)
        box = np.int0(box)

        # 计算高和宽
        height = abs(box[0][1] - box[2][1])
        # print(height)
        width = abs(box[0][0] - box[2][0])
        # print(width, height)

        # 筛选那些太细的矩形，留下扁的
        # if width < 8 or height < 8:
        #     continue

        # if height < 20 and width < 20:
        #     continue
        #
        if height > 100 and width > 100:
            continue

        region.append(box)

    return region


def detect(img):
    # 1.  转化成灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # h, w, _ = img.shape
    # 2. 形态学变换的预处理，得到可以查找矩形的图片
    dilation = preprocess(gray)

    # 3. 查找和筛选文字区域
    region = findTextRegion(dilation)

    return region


def text_detect(origin_img):
    ori_h, ori_w = origin_img.shape[:2]
    img = origin_img.copy()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # edges = cv2.Canny(gray, 0, 255)

    mser = cv2.MSER_create(min_area=0, max_area=600)
    regions, boxes = mser.detectRegions(gray)
    all_boxes = []
    for box in boxes:
        x, y, w, h = box
        # if w / h < 0.5:
        #     continue

        # if w / h > 10:
        #     continue
        # if h < 5:
        #     continue
        # if w > 0.8 * ori_w:
        #     continue

        # if h > 0.6 * ori_h:
        #     continue
        all_boxes.append([x, y, x + w, y + h])
        # cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), -1)

    # cv2.imshow('xx', img)
    # cv2.waitKey(0)

    return all_boxes

    # hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # lower = np.array([0, 0, 0])
    # upper = np.array([0, 0, 255])
    # mask = cv2.inRange(img, lower, upper)
    # res = cv2.bitwise_and(img, img, mask=mask)
    #
    # # cv2.imshow('res', res)
    # # cv2.waitKey(0)
    # #
    # boxes = []
    # region = detect(res)
    # for i in region:
    #     # print(i, i.shape)
    #     xs = []
    #     ys = []
    #     for j in i.tolist():
    #         xs.append(j[0])
    #         ys.append(j[1])
    #     xs.sort()
    #     ys.sort()
    #
    #     boxes.append([int(xs[0]), int(ys[0]), int(xs[-1]), int(ys[-1])])
    #
    # boxes.sort(key=lambda x: (x[1], x[0]))
    #
    # return NMS(boxes)


if __name__ == '__main__':

    data_dir = '/Users/yuemengrui/Data/RPAUI/web_cv_data/images的副本'
    label_dir = '/Users/yuemengrui/Data/RPAUI/web_cv_data/mser_labels'
    new_image_dir = '/Users/yuemengrui/Data/RPAUI/train_data_new/images'
    new_labels_dir = '/Users/yuemengrui/Data/RPAUI/train_data_new/labels'
    img_list = os.listdir(data_dir)

    for i in img_list:
        file_name = i.split('.')[0]
        img_path = os.path.join(data_dir, i)
        # img = cv2.imread(img_path)
        label_path = os.path.join(label_dir, file_name + '.png')
        label = cv2.imread(label_path, -1)

        num_1 = np.sum(label == 1)
        num_0 = np.sum(label == 0)

        r = num_1 / (num_0 + num_1)
        #
        if r > 0.02:
            new_image_path = os.path.join(new_image_dir, i)
            new_label_path = os.path.join(new_labels_dir, file_name + '.png')

            shutil.move(img_path, new_image_path)
            shutil.move(label_path, new_label_path)

            # print(file_name, r)
        #
        # cv2.imshow('ori', img)
        # cv2.waitKey(0)
        #
        # cv2.imshow('l', label*255)
        # cv2.waitKey(0)
        # boxes = text_detect(img)

        # h, w = img.shape[:2]
        # label = np.uint8(np.zeros((h, w)))
        # for box in boxes:
            # print('h: ', box[3] - box[1])
            # print('w: ', box[2] - box[0])
            # label[box[1]:box[3], box[0]:box[2]] = 1
            # cv2.rectangle(img, (box[0], box[1]), (box[2], box[3]), (0, 0, 255), -1)

        # cv2.imwrite(os.path.join(label_dir, file_name + '.png'), label)
        #
        # cv2.imshow('a', img)
        # cv2.waitKey(0)
