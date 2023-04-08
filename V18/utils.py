# *_*coding:utf-8 *_*
# @Author : YueMengRui
import requests

# temp_img_path = '/data/guorui/AI_Platform/Temp_files/train_temp/img.jpg'
# temp_target_path = '/data/guorui/AI_Platform/Temp_files/train_temp/target.jpg'

temp_img_path = './temp/img.jpg'
temp_target_path = './temp/target.jpg'


def get_token():
    data = {
        "acct": '18370652141',
        "password": '123456'
    }

    res = requests.post(url='https://bpa.aistreamrpa.com/authentication/login', json=data)
    token = res.json()['data']['token']
    return token


def upload_file(file_path, token):
    # token = get_token()
    url = 'https://bpa.aistreamrpa.com/file/upload/public?token={}'.format(token)

    data = {
        "file": open(file_path, 'rb')
    }

    res = requests.post(url, files=data)
    file_url = res.json()['data']['url']
    return file_url


def get_ocr_res(file_path, token):
    file_url = upload_file(file_path, token)

    url = 'http://134.175.246.119:9350/ai/ocr/general'

    req_data = {
        'url': file_url
    }

    res = requests.post(url=url, json=req_data)

    return res.json()['data']['results']

# token = get_token()
# resp = get_ocr_res('/Users/yuemengrui/Downloads/some_images/智能对话.jpg', token)
# print(resp)
# [{'box': [365, 11, 514, 15, 513, 50, 364, 47], 'text': ['知识库结构', 99]}, {'box': [21, 13, 171, 13, 171, 47, 21, 47], 'text': ['机器人类型', 99]}, {'box': [710, 15, 828, 15, 828, 45, 710, 45], 'text': ['核心技术', 99]}, {'box': [1054, 11, 1174, 15, 1173, 50, 1053, 46], 'text': ['落地难度', 99]}, {'box': [709, 74, 828, 78, 828, 108, 708, 105], 'text': ['信息检索', 99]}, {'box': [21, 79, 141, 79, 141, 108, 21, 108], 'text': ['FAQ-Bot', 99]}, {'box': [367, 79, 514, 79, 514, 108, 367, 108], 'text': ['{问题：答案}', 87]}, {'box': [1052, 76, 1083, 76, 1083, 110, 1052, 110], 'text': ['低', 99]}, {'box': [366, 139, 425, 139, 425, 171, 366, 171], 'text': ['文档', 99]}, {'box': [21, 140, 149, 140, 149, 169, 21, 169], 'text': ['MRC-Bot', 99]}, {'box': [713, 142, 965, 142, 965, 167, 713, 167], 'text': ['信息检索+机器阅读', 99]}, {'box': [1053, 137, 1084, 137, 1084, 172, 1053, 172], 'text': ['中', 99]}, {'box': [23, 203, 125, 203, 125, 232, 23, 232], 'text': ['KG-Bot', 99]}, {'box': [367, 203, 512, 203, 512, 232, 367, 232], 'text': ['知识三元组', 99]}, {'box': [713, 205, 955, 205, 955, 229, 713, 229], 'text': ['知识图谱构建/检索', 99]}, {'box': [1052, 200, 1084, 200, 1084, 234, 1052, 234], 'text': ['高', 99]}, {'box': [366, 263, 549, 263, 549, 292, 366, 292], 'text': ['槽位/对话策略', 99]}, {'box': [711, 263, 956, 263, 956, 292, 711, 292], 'text': ['对话状态跟踪/管理', 99]}, {'box': [23, 264, 147, 264, 147, 295, 23, 295], 'text': ['Task-Bot', 99]}, {'box': [1053, 262, 1084, 262, 1084, 297, 1053, 297], 'text': ['高', 99]}, {'box': [22, 324, 151, 328, 150, 363, 21, 359], 'text': ['Chat-Bot', 99]}, {'box': [709, 323, 828, 326, 828, 357, 708, 353], 'text': ['信息检索', 99]}, {'box': [369, 327, 544, 327, 544, 356, 369, 356], 'text': ['{寒喧语：回复}', 78]}, {'box': [1052, 324, 1084, 324, 1084, 359, 1052, 359], 'text': ['低', 99]}]
