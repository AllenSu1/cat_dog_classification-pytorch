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
fold_test = r'C:\Users\Allen\Desktop\Generate_Dataset\test'
classes = os.listdir(r'C:\Users\Allen\Desktop\Generate_Dataset\train')

test_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# model
model = torchvision.models.resnet101(pretrained=True, progress=True)
# 提取參數fc的輸入參數
fc_features = model.fc.in_features
# 將最後輸出類別改為5
model.fc = nn.Linear(fc_features, 5)
# 輸入訓練好權重
model.load_state_dict(torch.load(r"D:\GitHub\AllenSu1\ML\Mango_Classification\model\1.pth"))

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

D1 = []  #乳汁吸附
D2 = []  #機械傷害
D3 = []  #炭疽病
D4 = []  #著色不佳
D5 = []  #黑斑病


df = pd.read_csv(r'C:\Users\Allen\Desktop\Test.csv',
                encoding="utf8") 
imglist = np.array(df['image_id'])
for img in tqdm(imglist):
    image = Image.open(os.path.join(fold_test, img))
    r_image, class_list, _ = detect_image(image)
    # r_image.show()
    # print(class_list)
    D1.append(1) if 'class 1' in class_list else D1.append(0)
    D2.append(1) if 'class 2' in class_list else D2.append(0)
    D3.append(1) if 'class 3' in class_list else D3.append(0)
    D4.append(1) if 'class 4' in class_list else D4.append(0)
    D5.append(1) if 'class 5' in class_list else D5.append(0)

result_df = pd.DataFrame({
    'image_id': imglist,
    'D1': D1,
    'D2': D2,
    'D3': D3,
    'D4': D4,
    'D5': D5
})
# 輸出結果csv
result_df.to_csv(r'C:\Users\Allen\Desktop\Test1.csv', index=None)