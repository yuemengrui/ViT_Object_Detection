# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
import numpy as np
from model import ViT
from dataset import ViTSegDataset
from torch.utils.data import DataLoader


class Predictor(object):

    def __init__(self,
                 checkpoint_path='./checkpoints/model_best.pth',
                 device='cpu'):
        self.device = torch.device(device)
        self.model = ViT()

        self._load_checkpoint(checkpoint_path)

        self.model.to(self.device)
        self.model.eval()

    def _load_checkpoint(self, checkpoint_path):
        checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
        self.model.load_state_dict(checkpoint['state_dict'])

    def predict(self, img, img_boxes, img_texts, target, target_texts):
        img = img.to(self.device)
        img_boxes = img_boxes.to(self.device)
        img_texts = img_texts.to(self.device)
        target = target.to(self.device)
        target_texts = target_texts.to(self.device)

        with torch.no_grad():
            pred = self.model(img, img_boxes, img_texts, target, target_texts)

        return pred


if __name__ == '__main__':
    predictor = Predictor()
    dataset = ViTSegDataset(dataset_dir='/Users/yuemengrui/Data/RPA_UI/train_data_prebox')
    data_loader = DataLoader(dataset, batch_size=1, shuffle=True)
    for img, img_texts, img_boxes, target, target_texts, label_box in data_loader:
        print("====================")
        img_texts = img_texts.squeeze(dim=0)
        img_boxes = img_boxes.squeeze(dim=0)
        target_texts = target_texts.squeeze(dim=0)

        print('label_box: ', label_box)

        pred = predictor.predict(img, img_boxes, img_texts, target, target_texts)
        print(pred)
        break
