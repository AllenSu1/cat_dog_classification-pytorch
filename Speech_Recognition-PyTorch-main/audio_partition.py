import os, shutil, glob, random
# 定義資料集名稱
sets = ['train', 'val']
# 訓練集佔整體資料集比例
train_percent = 0.85
# 驗證集佔整體資料集比例
val_percent = 0.15
# 測試集佔整體資料集比例
test_percent = 0
# 原始資料集所在的路徑
original_dataset_dir = r'D:\dataset\ntut-ml-2020-spring-taiwanese-e2e\train_org'
# 用來儲存新資料集的位置
base_dir = r'D:\dataset\ntut-ml-2020-spring-taiwanese-e2e\train'
# 原始資料集相同的圖片名稱
image_name = ''

val_percent = val_percent / (val_percent + test_percent)
test_percent = 1
dataset_percent = [train_percent, val_percent, test_percent]

# 如果儲存資料集的資料夾不存在, 才建立資料夾
if not os.path.isdir(base_dir):
    os.mkdir(base_dir)

fileLists = os.listdir(original_dataset_dir)
# 分拆成訓練、驗證與測試資料夾
switch = True
for i in sets:
    set_path = os.path.join(base_dir, i)
    if not os.path.isdir(set_path):
        os.mkdir(set_path)
    # 查詢單一種類樣本數量
    if switch:
        dirPathPattern = os.path.join(original_dataset_dir,'*')
        sample = glob.glob(dirPathPattern)
        switch = False
    sample_number = int(len(sample) * dataset_percent[sets.index(i)])
    # 從資料集隨機抽樣
    filePaths = random.sample(sample, sample_number)
    sample = set(sample) - set(filePaths)
    sample = list(sample)
    # 將圖片複製到指定之資料夾
    for filePath in filePaths:
        # 讀取文件名稱
        fname = os.path.basename(filePath)
        src = os.path.join(original_dataset_dir, fname)
        dst = os.path.join(set_path, fname)
        shutil.copyfile(src, dst)
    print(i, len(os.listdir(set_path)))