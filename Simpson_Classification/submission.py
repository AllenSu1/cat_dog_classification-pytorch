import csv
import cv2
import os
from PIL import Image
from model import *
import torch
import numpy as np
import torchvision
import torch.nn as nn
from torchvision import datasets, models, transforms
import pandas as pd
from tqdm import tqdm

device = torch.device("cuda")

# 读入图片
image_path = 'D:\\Dataset\\simpson\\dataset\\test'
classes = os.listdir(r'D:\Dataset\simpson\preprocessing\test\train')

test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])


# model
model = torchvision.models.resnet101(pretrained=True, progress=True)
# 提取參數fc的輸入參數
fc_features = model.fc.in_features
# 將最後輸出類別改為20
model.fc = nn.Linear(fc_features, 20)
# 輸入訓練好權重
model.load_state_dict(torch.load("model/res101_unfrezee_StepLR_5_2.pth"))

# # 遷移學習 -> frezee
# for name, parameter in model.named_parameters():
#     # print(name)
#     if name == 'layer4.0.conv1.weight':
#         break
#     # if name == 'fc.weight':
#     #     break
#     parameter.requires_grad = False

model.to(device)
model.eval()

result = []
for i in tqdm(range(1, 991)):
    com_img = os.path.join(image_path, '%s.jpg' % i)
    test_data = Image.open(com_img).convert('RGB')
    data_transforms = test_transforms(test_data).to(device)
    pred = model(data_transforms[None, ...])
    predict_y = torch.max(pred, dim=1)[1]
    result.append(classes[int(predict_y)])
# print(result)
sample = pd.DataFrame({
    'id': pd.read_csv(r'D:\GitHub\t109318121\ML_Classification\sampleSubmission.csv').id,
    'character': result[:]
})

sample.to_csv('submission/res101_unfrezee_StepLR_5_2.csv', index=False)
