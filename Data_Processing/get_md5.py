# *_*coding:utf-8 *_*
# @Author : YueMengRui
import hashlib
import os
import shutil


def md5hex(data):
    m = hashlib.md5()
    m.update(data)
    return str(m.hexdigest())


img_dir = '/Users/yuemengrui/Data/RPAUI/SAP截图及样本数据/SAP界面截图'
new_dir = '/Users/yuemengrui/Data/RPAUI/sap/images'

img_list = os.listdir(img_dir)

for i in img_list:
    img_path = os.path.join(img_dir, i)

    with open(img_path, 'rb') as f:
        img_data = f.read()

    md5 = md5hex(img_data)

    new_path = os.path.join(new_dir, md5 + '.jpg')

    shutil.copy(img_path, new_path)

