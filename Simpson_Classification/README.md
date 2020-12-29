# classification-t109318121
##   目錄
1.  [說明](#做法說明)
2.  [模型訓練流程圖](#深度學習模型訓練流程圖)
3.  [訓練驗證損失](#訓練與驗證損失)
4.  [Loss分析改進](#Loss分析與改進)
5.  [結果](#測試結果)
##	說明
1. ##### 範例參考reference example @[LiDo @lido2370 PyTorch - 練習kaggle - Dogs vs. Cats - 使用自定義的 CNN model](https://hackmd.io/@lido2370/S1aX6e1nN?type=view)
2. ##### 友情贊助dataset_partition @[Chen-Yen Chang BlackyYen](https://github.com/BlackyYen)
3. ##### 頭貼鎮壓 @[Chen-Yen Chang BlackyYen](https://www.facebook.com/photo.php?fbid=857627608100415&set=a.111229042740279&type=3)

深度學習模型的建立和訓練，選擇框架與進行資料預處理，設定模型訓練前須選擇深度學習框架，此分類採用的第三方框架是Pytorch之核心庫支援，為跨平台的高級神經網路API深度神經網路，且非獨立的機器學習框架提供了更進階、更直觀的抽象集。此專案採用監督式學習，需要對訓練資料進行標記，為了確保訓練時模營不會產生過度擬合的情況，需要海量的資料，進行資料預處理，透過Data augmentation的資料擴增方法，藉由資料增強技術將20類人物各收集3000張圖像，總計60000張的資料集，其中訓練集、驗證集、測試集各占資料集的70%、15%、15%。
##	模型訓練流程圖
![深度學習模型訓練流程圖](https://github.com/AllenSu1/ML/blob/main/Simpson_Classification/FlowChart.jpg)
訓練模型使用自己所設計之cnn、ResNet101與ResNet151模型，所設計之cnn使用兩層convolution與兩層Maxpooling，並以ReLU為激勵函數，在下loss圖與混淆矩陣中可發現使用自己所設計之cnn在分類效果上不是很理想，故使用ResNet模型進行分類。

![cnn訓練與驗證loss圖](https://github.com/AllenSu1/ML/blob/main/Simpson_Classification/image/cnn__StepLR_5_cnn.jpg)
![cnn測試之混淆矩陣](https://github.com/AllenSu1/ML/blob/main/Simpson_Classification/image/cnn_StepLR_5.png)
ResNet在2015年被提出，在ImageNet比賽classification任務上獲得第一名，因此選擇ResNet101與ResNet151作為此研究的網路模型，在影像分類中，卷積神經網路最前面幾層用於辨識影像最基本的特徵，像是物體的紋理、輪廓、顏色等等，較後面的階層才是提取影像重要特徵的關鍵，因此保留CNN底層權重，僅針對後面階層與新的分類器進行訓練即可，一來可套用最好的權重進行學習，二來可以節省訓練時間。基於上述原因採用遷移學習呢，把已經訓練好的模型和權重直接納入到新的資料集當中進行訓練，只改變之前模型的分類器（全連線層和softmax），將分類輸出成20類，透過逐步解凍的技術，解凍'layer4.0.conv1.weight'後的階層，使模型訓練的效果達到最好。
### 逐步解凍步驟如下
####    1.凍結預訓練網路的卷積層權重
####    2.將舊的全連線層，換上新的欲輸出種類個數的全連線層與分類器
####    3.解凍部分頂部的卷積層，保留底部卷積神經網路的權重
####    4.對解凍的卷積層與全連線層進行訓練，得到新的權重
##	訓練驗證損失
####    ResNet101
ResNet101超參數設計：設計learning_rate = 0.0001，EPOCH = 100，batch_size = 180，val_batch_size = 100，step_size=5。
訓練中使用學習率下降，以利找到局部最小值，在訓練的過程中，可以看到使用已訓練好的模型進行訓練，其loss很快就下降到0.5，再逐步收斂至趨近於0。

![ResNet101訓練與驗證loss圖](https://github.com/AllenSu1/ML/blob/main/Simpson_Classification/image/res101_frezee.png)
####    ResNet151
ResNet151超參數設計：設計learning_rate = 0.0001，EPOCH = 10，batch_size = 180，val_batch_size = 100，step_size=5。
比較ResNet151訓練過程中未遷移學習，但兩者在學習成效上準確率皆非常好，但所需時間卻大大拉長，故僅以10epoch進行測試。

![ResNet151訓練與驗證loss圖](https://github.com/AllenSu1/ML/blob/main/Simpson_Classification/image/res151_unfrezee_StepLR_5.jpg)
##   Loss分析改進
在資料預處理的過程中，比較"先將資料集進行擴增再進行分割"與"先將資料集進行分割再進行擴增"可能會有不同效果，此次專案是採用先將資料集擴增再分割，再預處理的過程中可能也會有過度擬合的問題發生。再訓練上，或許參數還可以調整更好，以提升訓練成效。
##  結果
ResNet101
以每類375張圖像，共7500張圖像計行測試，其ResNet101測試結果以混淆矩陣呈現如下，其測試準確率達99.27%。

![ResNet101測試之混淆矩陣](https://github.com/AllenSu1/ML/blob/main/Simpson_Classification/image/res101_frezee_cm.png)
ResNet151
以每類375張圖像，共7500張圖像計行測試，其ResNet151測試結果以混淆矩陣呈現如下，其測試準確率達99.66%。

![ResNet151測試之混淆矩陣](https://github.com/AllenSu1/ML/blob/main/Simpson_Classification/image/res151_unfrezee_StepLR_5.png)

