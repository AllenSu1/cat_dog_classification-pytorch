import pandas as pd
import numpy as np
import os
import cv2
from tqdm import tqdm

# 讀取資料集標籤檔
df = pd.read_csv(r"C:\Users\Allen\Desktop\mango\stage3\Test_mangoXYWH.csv",encoding="utf8", header=None)
path =r"C:\Users\Allen\Desktop\mango\stage3\Test"
picture_num = 0
# 逐列遍歷dataframe
with tqdm(total=len(df)) as pbar:
    for index, row in df.iterrows():
        data_row = df.loc[index].values[0:-1] #讀入每一行數值
        file_name = data_row[0] #第0行為檔案圖片名稱
        col = 1
        count = 0
        # print(file_name)
        # 將一顆芒果中多個瑕疵部位分開裁減
        try:
            while not np.isnan(data_row[col]):
                count = count + 1
                data = np.array(data_row[col:col+4], dtype=int)
                label = data_row[col+4]
                col = col + 5
                # 讀取圖檔
                img = cv2.imread(os.path.join(path, file_name))
                # 裁切圖片
                x, y, w, h = data
                # print(x,y,w,h)
                crop_img = img[ y:y+h, x:x+w ]
                # # 顯示圖片
                # cv2.imshow("crop_img", crop_img)
                # cv2.waitKey(0)
                # 根據類別寫入裁剪後的圖片
                savepath = os.path.join(r'C:\Users\Allen\Desktop\Generate_Dataset\test', label)
                if not os.path.isdir(savepath):
                    os.mkdir(savepath)
                savepath = os.path.join(savepath, file_name[:-4]) + '_' + str( count) +'.jpg'
                cv_img=cv2.imencode('.jpg', crop_img)[1].tofile(savepath)
                picture_num = picture_num + 1
        except Exception as e:
            print(file_name, '發生錯誤\n', e)
        # 更新進度條
        pbar.update(1)
        pbar.set_description('Valid_Dataset')
        pbar.set_postfix(
                **{
                    'File_name' : file_name,
                    'Picture_num' : picture_num
                })