import torch
import numpy as np
import pandas as pd
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler, MinMaxScaler
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# 設定 GPU
device = torch.device("cuda")

# 讀取 csv 檔案
train_data = pd.read_csv('./ntut-ml-regression-2020/train-v3.csv')
val_data = pd.read_csv('./ntut-ml-regression-2020/valid-v3.csv')
test_data = pd.read_csv('./ntut-ml-regression-2020/test-v3.csv')

# 將 train_data, val_data, test_data 合併
all_features = pd.concat(
    (train_data.iloc[:, 2:], val_data.iloc[:, 2:], test_data.iloc[:, 1:]))

# one-hot
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
numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
all_features[numeric_features] = all_features[numeric_features].apply(
    lambda x: (x - x.mean()) / (x.std()))

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

# 房價預測結果寫入csv


def test_model(model, test_features):
    model = torch.load(model)
    # 使用 GPU 測試
    model.to(device)
    with torch.no_grad():
        predictions = model(test_features).detach().cpu().numpy()
        my_submission = pd.DataFrame({
            'id':
            pd.read_csv(
                r'./ntut-ml-regression-2020/test-v3.csv').id,
            'price':
            predictions[:, 0]
        })
        my_submission.to_csv('{}.csv'.format('./result/blacky_36758'),
                             index=False)


def main():
    test_model('./log/model.pth', test_features)


if __name__ == "__main__":
    main()

# %%
