# *_*coding:utf-8 *_*
# @Author : YueMengRui
import requests
import cv2
import numpy as np


def get_ocr(data):
    url = 'http://134.175.246.119:9350/ai/ocr/byte'

    req_data = {
        'file': data
    }

    resp = requests.post(url=url, files=req_data)
    print(resp.json())
    return resp.json()['data']['results']


img = cv2.imread('/Users/yuemengrui/Data/OCR/train_data/cn/cn00000012.jpg')

img_encode = cv2.imencode('.jpg', img)[1]

data_encode = np.array(img_encode)

get_ocr(data_encode)
