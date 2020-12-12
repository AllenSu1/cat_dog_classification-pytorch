from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
from net.model import Net
from sklearn.decomposition import PCA
from torch.optim import lr_scheduler
import copy
from time import sleep
from tqdm import tqdm
import matplotlib.pyplot as plt
import sys
import pandas as pd
import numpy as np
import torch.nn as nn
from IPython import display
import torch.utils.data
import torch
import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Hyperparameter（超參數）
batch_size = 64
learning_rate = 1e-5
num_epochs = 1000

# 設定 GPU
device = torch.device("cuda")

# 讀取 csv 檔案
train_data = pd.read_csv('./ntut-ml-regression-2020/train-v3.csv')
val_data = pd.read_csv('./ntut-ml-regression-2020/valid-v3.csv')
test_data = pd.read_csv('./ntut-ml-regression-2020/test-v3.csv')

# 將 train_data, val_data, test_data 合併
all_features = pd.concat(
    (train_data.iloc[:, 2:], val_data.iloc[:, 2:], test_data.iloc[:, 1:]))

cate_feature = ['sale_yr', 'sale_month', 'sale_day']
for item in cate_feature:
    all_features[item] = LabelEncoder().fit_transform(all_features[item])
    item_dummies = pd.get_dummies(all_features[item])
    item_dummies.columns = [
        item + str(i + 1) for i in range(item_dummies.shape[1])
    ]
    all_features = pd.concat([all_features, item_dummies], axis=1)

all_features.drop(cate_feature, axis=1, inplace=True)

# 正規化
# numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
# all_features[numeric_features] = all_features[numeric_features].apply(
#     lambda x: (x - x.mean()) / (x.std()))

# normalized
scaler = StandardScaler()
all_features = scaler.fit_transform(all_features)
all_features = pd.DataFrame(all_features)

# normalized
# scaler = MinMaxScaler()
# all_features = scaler.fit_transform(all_features)
# all_features = pd.DataFrame(all_features)

# print(all_features)

# 獲取每筆資料的數量
n_train = train_data.shape[0]
n_val = val_data.shape[0]
n_val = n_train + n_val

# 切割資料
train_features = torch.tensor(all_features[:n_train].values,
                              dtype=torch.float).to(device)
val_features = torch.tensor(all_features[n_train:n_val].values,
                            dtype=torch.float).to(device)
test_features = torch.tensor(all_features[n_val:].values,
                             dtype=torch.float).to(device)
train_labels = torch.tensor(train_data.price.values,
                            dtype=torch.float).view(-1, 1).to(device)
val_labels = torch.tensor(val_data.price.values,
                          dtype=torch.float).view(-1, 1).to(device)
# print(train_features)

weight_decay = 0
train_ls, val_ls = [], []

# 切割訓練集
train_dataset = torch.utils.data.TensorDataset(train_features, train_labels)
train_iter = torch.utils.data.DataLoader(train_dataset,
                                         batch_size,
                                         shuffle=True)
val_dataset = torch.utils.data.TensorDataset(val_features, val_labels)
val_iter = torch.utils.data.DataLoader(val_dataset, batch_size, shuffle=True)

# 呼叫模型
model = Net(features=train_features.shape[1])
# 使用 GPU 訓練
model.to(device)
# 定義損失函數
# criterion = nn.MSELoss(reduction='mean')
criterion = nn.L1Loss(reduction='mean')
criterion = criterion.to(device)
# 定義優化器
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=learning_rate,
                             weight_decay=weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)
model = model.float()
switch = True
# 訓練模型
for epoch in range(num_epochs):
    scheduler.step()
    print('Epoch {}/{}'.format(epoch + 1, num_epochs))
    for X, y in tqdm(train_iter):
        loss = criterion(model(X.float()), y.float())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    print('Loss: {:.0f}'.format(loss))
    for X, y in val_iter:
        val_loss = criterion(model(X.float()), y.float())
    train_ls.append(loss)
    val_ls.append(val_loss)
    print('val_loss: {:.0f}'.format(val_loss))

    # 複製最好的模型參數資料
    if switch == True:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
        switch = False
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        best_model_wts = copy.deepcopy(model.state_dict())
    print('best_val_loss: {:.0f}'.format(best_val_loss))
    # model.param_groups[0]["lr"] = 0.004
    # 輸出學習率
    # print(optimizer.state_dict()['param_groups'][0]['lr'])

# 讀取最好的權重
model.load_state_dict(best_model_wts)
# 儲存模型與權重
torch.save(model, './log/model.pth')

# 繪製圖
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_ls, label='train_losses')
plt.plot(val_ls, label='val_losses')
plt.legend(loc='best')
plt.savefig('d:/GitHub/AllenSu1/blackyregression/image/image.jpg')
plt.show()
