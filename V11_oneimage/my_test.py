# *_*coding:utf-8 *_*
# @Author : YueMengRui
import torch
from model import ViT

model = ViT()

checkpoint = torch.load('./checkpoints/099e14db912be663fb7592ed61819b2f.pth', map_location=torch.device('cpu'))
model.load_state_dict(checkpoint['state_dict'])

# torch.save(model.state_dict(), './checkpoints/state_dict.pth')

torch.save(model, './checkpoints/model.pth')
