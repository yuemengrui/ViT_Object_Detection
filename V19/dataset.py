# *_*coding:utf-8 *_*
# @Author : YueMengRui
import os
import numpy as np
from torch.utils.data import Dataset
import cv2
import torch
import random
import time
import json
from torchvision import transforms
from torch.utils.data import DataLoader
from utils import get_token, get_ocr_res, temp_img_path, temp_target_path


class ImageRandomScaleResize(object):

    def __init__(self):
        self.w_h_scale = [1.25, 1.33, 1.37, 1.6, 1.667, 1.778, 2.13]
        self.screen_scale = [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0]

    def __call__(self, img, box):
        h, w = img.shape[:2]

        w_h_scale = random.choice(self.w_h_scale)

        if w / h > w_h_scale:
            new_w = w
            new_h = w / w_h_scale

        elif w / h < w_h_scale:
            new_w = h * w_h_scale
            new_h = h
        else:
            new_w = w
            new_h = h

        screen_scale = random.choice(self.screen_scale)

        new_w = int(new_w * screen_scale)
        new_h = int(new_h * screen_scale)

        w_scale = new_w / w
        h_scale = new_h / h
        img = cv2.resize(img, (new_w, new_h))
        x1, y1, x2, y2 = int(box[0] * w_scale), int(box[1] * h_scale), int(box[2] * w_scale), int(box[3] * h_scale)

        return img, [x1, y1, x2, y2]


class ImageResize(object):

    def __init__(self, size_range):
        self.size_range = size_range
        # self.padding = padding

    def __call__(self, img, box):
        h, w = img.shape[:2]
        # if h > w:
        #     new_w = h
        #     border = int((new_w - w) / 2)
        #     img = cv2.copyMakeBorder(img, 0, 0, border, border, cv2.BORDER_CONSTANT, value=self.padding)
        # else:
        #     new_h = w
        #     border = int((new_h - h) / 2)
        #     img = cv2.copyMakeBorder(img, border, border, 0, 0, cv2.BORDER_CONSTANT, value=self.padding)

        w_scale = self.size_range[1] / w
        h_scale = self.size_range[0] / h
        img = cv2.resize(img, (self.size_range[1], self.size_range[0]))
        x1, y1, x2, y2 = int(box[0] * w_scale), int(box[1] * h_scale), int(box[2] * w_scale), int(box[3] * h_scale)

        return img, [x1, y1, x2, y2], h_scale, w_scale


class TargetResize(object):
    def __init__(self, size_range):
        self.size_range = size_range

    def __call__(self, img):
        # h, w = img.shape[:2]
        #
        # if h > self.size or w > self.size:
        #     scale = self.size / max(h, w)
        #     new_h, new_w = int(h * scale), int(w * scale)
        #
        #     img = cv2.resize(img, (new_w, new_h))
        #
        # h_border = int((self.size - h) / 2) if int((self.size - h) / 2) > 0 else 0
        # w_border = int((self.size - w) / 2) if int((self.size - w) / 2) > 0 else 0
        # if h_border != 0 or w_border != 0:
        #     img = cv2.copyMakeBorder(img, h_border, h_border, w_border, w_border, cv2.BORDER_CONSTANT, value=0)
        #
        # # x1, y1, x2, y2 = w_border, h_border, w_border + w, h_border + h
        # nh, nw = img.shape[:2]
        # if nh != self.size or nw != self.size:
        #     # h_scale = self.size / nh
        #     # w_scale = self.size / nw
        #     img = cv2.resize(img, (self.size, self.size))
        #     # x1, y1, x2, y2 = int(w_border * w_scale), int(h_border * h_scale), int((w_border + w) * w_scale), int(
        #     #     (h_border + h) * h_scale)
        img = cv2.resize(img, (self.size_range[1], self.size_range[0]))

        return img


class ViTSegDataset(Dataset):

    def __init__(self, dataset_dir, mode='train', **kwargs):

        self.image_resize = ImageResize(size_range=(512, 832))
        self.target_resize = TargetResize(size_range=(32, 64))
        self.image_random_scale_resize = ImageRandomScaleResize()

        self.transform = transforms.Compose([transforms.ToTensor(),
                                             transforms.Normalize((0.866, 0.873, 0.895), (0.278, 0.254, 0.224)),
                                             ])

        self.data_list = self._load_data(dataset_dir, mode)

        self.token = get_token()
        self.token_count = 0

        self.char_dict = self.get_character()

    def get_character(self):
        character = " "
        with open('./ch.txt', "rb") as fin:
            lines = fin.readlines()
            for line in lines:
                line = line.decode('utf-8').strip("\n").strip("\r\n")
                character += line

        char_dict = {}
        for i, char in enumerate(character):
            char_dict[char] = i

        return char_dict

    def _text_encode(self, text):
        text_encode = []
        if text == '':
            text_encode.append(self.char_dict[" "])
        else:
            for t in text.upper():
                if t not in self.char_dict.keys():
                    text_encode.append(self.char_dict[" "])
                else:
                    text_encode.append(self.char_dict[t])

        # if len(text_encode) > max_length:
        #     text_encode = text_encode[:max_length]
        # else:
        #     pad = max_length - len(text_encode)
        #     for _ in range(pad):
        #         text_encode.append(self.char_dict[" "])

        return text_encode

    def _load_data(self, dataset_dir, mode='train'):
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

        if mode == 'val':
            return data_list[100:200]

        return data_list

    def split_words_box(self, text, box):
        cxs = []
        x1, y1, x2, y2 = box
        cy = int(y1 + (y2 - y1) / 2)
        cys = [cy] * len(text)
        unit_w = (x2 - x1) / len(text)
        for i in range(len(text)):
            cxs.append(x1 + unit_w * i + unit_w / 2)

        assert len(cxs) == len(text) == len(cys), 'text length:{} should == cxs:{} == cys:{}'.format(len(text),
                                                                                                     len(cxs),
                                                                                                     len(cys))

        return cxs, cys

    def ocr_res_handler(self, h_scale, w_scale):
        if self.token_count > 600:
            self.token = get_token()
            self.token_count = 0

        self.token_count += 1

        img_ocr_res = get_ocr_res(temp_img_path, self.token)
        target_ocr_res = get_ocr_res(temp_target_path, self.token)

        img_texts = []
        center_xs = []
        center_ys = []
        scale = np.array((w_scale, h_scale))
        for i in img_ocr_res:
            if i['text'][1] > 72:
                box = np.array(i['box']).reshape((4, 2))
                norm_box = box * scale
                x1 = int(min(norm_box[:, 0]))
                x2 = int(max(norm_box[:, 0]))
                y1 = int(min(norm_box[:, 1]))
                y2 = int(max(norm_box[:, 1]))
                cxs, cys = self.split_words_box(i['text'][0], [x1, y1, x2, y2])

                temp_w_list = []
                temp_cxs = []
                temp_cys = []
                for n in range(len(i['text'][0])):
                    if i['text'][0][n].isalnum or (u'\u4e00' <= i['text'][0][n] <= u'\u9fff'):
                        temp_w_list.append(i['text'][0][n])
                        temp_cxs.append(cxs[n])
                        temp_cys.append(cys[n])

                img_text_encode = self._text_encode(''.join(temp_w_list))
                img_texts.extend(img_text_encode)
                center_xs.extend(temp_cxs)
                center_ys.extend(temp_cys)

        assert len(img_texts) == len(center_xs) == len(
            center_ys), 'img_text len:{} should == center_xs:{} == center_ys:{}'.format(len(img_texts), len(center_xs),
                                                                                        len(center_ys))

        if len(img_texts) > 4096:
            img_texts = img_texts[:4096]
            center_xs = center_xs[:4096]
            center_ys = center_ys[:4096]

        if len(img_texts) == 0:
            img_text_encode = self._text_encode(' ')
            img_texts.extend(img_text_encode)
            center_xs.append(0)
            center_ys.append(0)

        target_texts = []
        for j in target_ocr_res:
            if j['text'][1] > 72:
                temp_w_list = []
                for n in range(len(j['text'][0])):
                    if j['text'][0][n].isalnum or (u'\u4e00' <= j['text'][0][n] <= u'\u9fff'):
                        temp_w_list.append(j['text'][0][n])

                target_text_encode = self._text_encode(''.join(temp_w_list))
                target_texts.extend(target_text_encode)

        if len(target_texts) == 0:
            target_text_encode = self._text_encode(' ')
            target_texts.extend(target_text_encode)

        return np.array(img_texts), np.array(center_xs), np.array(center_ys), np.array(target_texts)

    def __getitem__(self, idx):
        try:
            data = self.data_list[idx]

            img = cv2.imread(data['img_path'])
            all_boxes = data['boxes']

            target_box = all_boxes[random.randint(0, len(all_boxes) - 1)]

            target = img[target_box[1]:target_box[3], target_box[0]:target_box[2]]

            img, target_box = self.image_random_scale_resize(img, target_box)

            # img_h, img_w = img.shape[:2]
            cv2.imwrite(temp_img_path, img)
            cv2.imwrite(temp_target_path, target)

            img, target_box, h_scale, w_scale = self.image_resize(img, target_box)
            im_h, im_w = img.shape[:2]
            target = self.target_resize(target)

            img_texts, c_xs, c_ys, target_texts, = self.ocr_res_handler(h_scale, w_scale)

            # label_box = [target_box[0] / im_w, target_box[1] / im_h, target_box[2] / im_w, target_box[3] / im_h]

            target_cx = target_box[0] + (target_box[2] - target_box[0]) / 2
            target_cy = target_box[1] + (target_box[3] - target_box[1]) / 2
            target_h = target_box[3] - target_box[1]
            target_w = target_box[2] - target_box[0]
            label_box = [target_cx / im_w, target_cy / im_h, target_w / im_w, target_h / im_h]

            img = self.transform(img)
            target = self.transform(target)
            label_box = torch.from_numpy(np.array(label_box))

            # return img, target, label_box, img_texts, c_xs, c_ys
            return img, torch.from_numpy(img_texts).int(), torch.from_numpy(c_xs).int(), torch.from_numpy(
                c_ys).int(), target, torch.from_numpy(target_texts).int(), label_box

        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(self.__len__()))

    def __len__(self):
        return len(self.data_list)


if __name__ == '__main__':
    import torch.nn.functional as F
    from loss import generalized_box_iou, box_iou, box_cxcywh_to_xyxy

    from model import ViT

    net = ViT()
    #     # num_1 = 0
    #     # num_0 = 0
    #
    dataset = ViTSegDataset(dataset_dir='/Users/yuemengrui/Data/RPA_UI/train_data_prebox')
    # img, target, label_box, img_texts, c_xs, c_ys = dataset[1]
    # print(len(img_texts))
    # print(len(c_xs))
    # print(len(c_ys))
    # cv2.imshow('im', img)
    # cv2.waitKey(0)
    # cv2.imshow('t', target)
    # cv2.waitKey(0)
    # print(label_box)
    # h, w = img.shape[:2]
    # print(h, w)
    # cx = label_box[0] * w
    # cy = label_box[1] * h
    # t_w = label_box[2] * w
    # t_h = label_box[3] * h
    # print(cx, cy, t_w, t_h)
    # x1 = int(cx - t_w / 2)
    # y1 = int(cy - t_h / 2)
    # x2 = int(cx + t_w / 2)
    # y2 = int(cy + t_h / 2)
    # print(x1,y1,x2,y2)
    # cv2.rectangle(img, (x1,y1), (x2,y2), (0,0,255), 2)
    # cv2.imshow('xx', img)
    # cv2.waitKey(0)

    #
    # for i in range(len(img_texts)):
    #     print(img_texts[i])
    #     print(img_boxes[i])
    #     cv2.rectangle(img, (int(img_boxes[i][0]), int(img_boxes[i][1])), (int(img_boxes[i][2]), int(img_boxes[i][3])),
    #                   (0, 0, 255), 2)
    #     cv2.imshow('xx', img)
    #     cv2.waitKey(0)
    #     # for i in range(len(dataset)):
    #     #     print(i)
    #     #     label = dataset[i]
    #     #     num_1 += np.sum(label == 1)
    #     #     num_0 += np.sum(label == 0)
    #     #
    #     # all = 512*832*367
    #     # n_1 = num_1 / all
    #     # n_0 = num_0 / all
    #     # w_1 = 0.5 / n_1
    #     # w_0 = 0.5 / n_0
    #     # print(num_0)
    #     # print(num_1)
    #     # print(all)
    #     # print(n_1)
    #     # print(n_0)
    #     # print('w_1: ', w_1)
    #     # print('w_0: ', w_0)
    #     #     dataset[i]
    train_loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True)

    for img, img_texts, cxs, cys, target, target_texts, label_box in train_loader:
        print("====================")
        print('img: ', img.shape)
        print('img_texts: ', img_texts.shape)
        print('cxs: ', cxs.shape)
        print('cys: ', cys.shape)
        print('target: ', target.shape)
        print('target_text: ', target_texts.shape)
        print(label_box.shape, label_box)
        s = time.time()
        out = net(img, img_texts, cxs, cys, target, target_texts)
        print(out)
        print(time.time() - s)
        break

        # label: [0.5643, 0.5029, 0.0805, 0.1777]
        # pred: [0.4972, 0.5460, 0.4496, 0.4999]
        # iou: [0.0637]


