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

    def __call__(self, img, label):
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

        # w_scale = new_w / w
        # h_scale = new_h / h
        img = cv2.resize(img, (new_w, new_h))
        label = cv2.resize(label, (new_w, new_h))
        # x1, y1, x2, y2 = int(box[0] * w_scale), int(box[1] * h_scale), int(box[2] * w_scale), int(box[3] * h_scale)

        return img, label


class ImageResize(object):

    def __init__(self, size_range):
        self.size_range = size_range
        # self.padding = padding

    def __call__(self, img, label):
        # h, w = img.shape[:2]
        # if h > w:
        #     new_w = h
        #     border = int((new_w - w) / 2)
        #     img = cv2.copyMakeBorder(img, 0, 0, border, border, cv2.BORDER_CONSTANT, value=self.padding)
        # else:
        #     new_h = w
        #     border = int((new_h - h) / 2)
        #     img = cv2.copyMakeBorder(img, border, border, 0, 0, cv2.BORDER_CONSTANT, value=self.padding)

        # w_scale = self.size_range[1] / w
        # h_scale = self.size_range[0] / h
        img = cv2.resize(img, (self.size_range[1], self.size_range[0]))
        label = cv2.resize(label, (self.size_range[1], self.size_range[0]))
        # x1, y1, x2, y2 = int(box[0] * w_scale), int(box[1] * h_scale), int(box[2] * w_scale), int(box[3] * h_scale)

        return img, label


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
        return np.array(text_encode)

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

    def ocr_res_handler(self):
        if self.token_count > 600:
            self.token = get_token()
            self.token_count = 0

        self.token_count += 1

        img_ocr_res = get_ocr_res(temp_img_path, self.token)
        target_ocr_res = get_ocr_res(temp_target_path, self.token)

        img_texts = []
        img_text_lengths = []
        img_boxes = []
        for i in img_ocr_res:
            if i['text'][1] > 72:
                img_text_encode = self._text_encode(i['text'][0])
                img_text_lengths.append(img_text_encode.shape[0])
                img_texts.append(img_text_encode)
                img_boxes.append([int(x) for x in i['box']])
        if len(img_texts) == 0:
            img_text_encode = self._text_encode(' ')
            img_text_lengths.append(img_text_encode.shape[0])
            img_texts.append(img_text_encode)
            img_boxes.append([0, 0, 0, 0, 0, 0, 0, 0])
            img_boxes = np.array(img_boxes)
        else:
            img_boxes = np.array(img_boxes)
            img_boxes_min = img_boxes.min(0)
            img_boxes_max = img_boxes.max(0)
            img_boxes = (img_boxes - img_boxes_min) / (img_boxes_max - img_boxes_min)
            img_boxes = (img_boxes - 0.5) / 0.5

        target_texts = []
        target_text_lengths = []
        for j in target_ocr_res:
            if j['text'][1] > 72:
                target_text_encode = self._text_encode(j['text'][0])
                target_text_lengths.append(target_text_encode.shape[0])
                target_texts.append(target_text_encode)
        if len(target_texts) == 0:
            target_text_encode = self._text_encode(' ')
            target_text_lengths.append(target_text_encode.shape[0])
            target_texts.append(target_text_encode)

        return np.array(img_texts), np.array(img_text_lengths), img_boxes, np.array(target_texts), np.array(
            target_text_lengths)

    def __getitem__(self, idx):
        try:
            data = self.data_list[idx]

            img = cv2.imread(data['img_path'])
            all_boxes = data['boxes']

            ori_h, ori_w = img.shape[:2]

            label = np.uint8(np.zeros((ori_h, ori_w)))

            target_box = all_boxes[random.randint(0, len(all_boxes) - 1)]

            target = img[target_box[1]:target_box[3], target_box[0]:target_box[2]]

            label[target_box[1]:target_box[3], target_box[0]:target_box[2]] = 1

            img, label = self.image_random_scale_resize(img, label)

            img, label = self.image_resize(img, label)
            cv2.imwrite(temp_img_path, img)

            target = self.target_resize(target)
            cv2.imwrite(temp_target_path, target)

            img_texts, img_text_lengths, img_boxes, target_texts, target_text_lengths = self.ocr_res_handler()

            img = self.transform(img).unsqueeze(0)
            target = self.transform(target).unsqueeze(0)
            label = torch.from_numpy(label).unsqueeze(0)

            return img, img_texts, img_text_lengths, img_boxes, target, target_texts, target_text_lengths, label

        except Exception as e:
            print(e)
            return self.__getitem__(np.random.randint(self.__len__()))

    def collate(self, samples):
        img, img_texts, img_text_lengths, img_boxes, target, target_texts, target_text_lengths, label = map(list,
                                                                                                            zip(*samples))
        img = torch.cat(img, dim=0)
        target = torch.cat(target, dim=0)
        label = torch.cat(label, dim=0)
        img_boxes = np.concatenate(img_boxes)
        img_text_lengths = np.concatenate(img_text_lengths)
        target_text_lengths = np.concatenate(target_text_lengths)

        max_text_length = max(img_text_lengths.max(), target_text_lengths.max())

        img_texts = np.concatenate(img_texts)

        im_text = [np.expand_dims(np.pad(t, (0, max_text_length - t.shape[0]), 'constant'), axis=0) for t in img_texts]
        img_texts = np.concatenate(im_text)

        target_texts = np.concatenate(target_texts)

        tar_text = [np.expand_dims(np.pad(t, (0, max_text_length - t.shape[0]), 'constant'), axis=0) for t in
                    target_texts]
        target_texts = np.concatenate(tar_text)

        return img, torch.from_numpy(img_texts), torch.from_numpy(img_text_lengths), torch.from_numpy(
            img_boxes).float(), \
            target, torch.from_numpy(target_texts), torch.from_numpy(target_text_lengths), label

    def __len__(self):
        return len(self.data_list)


# if __name__ == '__main__':
#     from model import ViT
#
#     net = ViT()
#
#     dataset = ViTSegDataset(dataset_dir='/Users/yuemengrui/Data/RPA_UI/train_data_prebox')
#     train_loader = DataLoader(dataset, batch_size=1, shuffle=True, drop_last=True, collate_fn=dataset.collate)
#
#     for img, img_texts, img_text_lengths, img_boxes, target, target_texts, target_text_lengths, label in train_loader:
#         print(img.shape)
#         print(img_texts.shape)
#         print(img_text_lengths.shape)
#         print(img_boxes.shape)
#         print(target.shape)
#         print(target_texts.shape)
#         print(target_text_lengths.shape)
#         print(label.shape)
#
#         s = time.time()
#         out = net(img, img_boxes, img_texts, img_text_lengths, target, target_texts, target_text_lengths)
#         print(time.time() - s)
#         break
