import pandas as pd
import numpy as np
import os
from tqdm import tqdm

csv_path = r'D:\dataset\mango\step_2\dev.csv'
folder_path = r'D:\dataset\mango\step_2\Dev'

save_path = r'./train_val_txt/val.txt'
file_path = r'D:/dataset/mango/step_2/Dev'


# 寫入txt
def data2txt(file, data, label):
    x, y, w, h = data
    box = '%s,%s,%s,%s,%s' % (x, y, x + w, y + h, label)
    f.write(box)


lable_dict = {
    '不良-乳汁吸附': 0,
    '不良-炭疽病': 1,
    '不良-著色不佳': 2,
    '不良-黑斑病': 3,
    '不良-機械傷害': 4
}  #Label Encoder
f = open(save_path, 'w')  #寫入txt檔案位置

df = pd.read_csv(csv_path, encoding="utf8", header=None)  #讀取metadata
with tqdm(total=len(df)) as pbar:
    for index, row in df.iterrows():
        data_row = df.loc[index].values[0:-1]  #讀入每一行數值
        file_name = data_row[0]  #第0行為檔案圖片名稱
        f.write(file_path + '/' + file_name)  #兩個框座標中間的空白
        col = 1

        try:
            while not np.isnan(data_row[col]):
                f.write(' ')
                data = np.array(data_row[col:col + 4], dtype=int)
                label = data_row[col + 4]
                label = lable_dict[label]  #Label Encoding
                data2txt(f, data, label)
                col = col + 5  #讀取下一個瑕疵座標
            f.write('\n')  #換行後繼續寫入下一張圖片
        # 讀取最長的那一列會超出csv範圍
        except Exception as e:
            print(file_name, '發生錯誤\n', e)
        # 更新進度條
        pbar.update(1)
        pbar.set_description('train_txt')
        pbar.set_postfix(**{
            'File_name': file_name,
        })
f.close()