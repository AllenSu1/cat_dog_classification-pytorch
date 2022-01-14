# 貓狗分類 cat_dog_classification-pytorch
cat_dog_classification-pytorch
##   目錄
1.  [說明](#做法說明)
2.  [訓練驗證設計](#訓練驗證設計)
3.  [Loss分析改進](#Loss分析與改進)
4.  [結果](#測試結果)
##	說明
##### 範例參考reference example @[LiDo @lido2370 PyTorch - 練習kaggle - Dogs vs. Cats - 使用自定義的 CNN model](https://hackmd.io/@lido2370/S1aX6e1nN?type=view)
##	訓練驗證設計
####    ResNet18
ResNet18超參數設計：設計learning_rate = 0.00001，EPOCH = 10，batch_size = 16，val_batch_size = 8。

##   Loss分析改進
在資料預處理的過程中，比較"先將資料集進行擴增再進行分割"與"先將資料集進行分割再進行擴增"可能會有不同效果，此次專案是採用先將資料集擴增再分割，再預處理的過程中可能也會有過度擬合的問題發生。再訓練上，或許參數還可以調整更好，以提升訓練成效。
##  結果
ResNet18分類結果
以每類500張圖像，共1000張圖像計行測試，其ResNet18測試結果以混淆矩陣呈現如下，其測試準確率達99%。

![ResNet18測試之混淆矩陣](https://github.com/AllenSu1/cat_dog_classification-pytorch/blob/main/confusion_matrix.png)


