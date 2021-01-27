import torch
import torchvision
import numpy as np
import torch.nn as nn
import matplotlib.pyplot as plt
from tqdm import tqdm
from pathlib import Path
# from model import CNN_Model
# from ResNet18 import resnet18
from torch.optim import lr_scheduler
from torchvision import datasets, models, transforms
from torch.utils.data.sampler import SubsetRandomSampler

train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA不可用，使用CPU訓練')
else:
    print('CUDA可用，使用GPU訓練')

PATH_train = "D:/Dataset/simpson/preprocessing/train"
PATH_val = "D:/Dataset/simpson/preprocessing/val"

TRAIN = Path(PATH_train)
VALID = Path(PATH_val)

# number of subprocesses to use for data loading
num_workers = 0
# 設計超參數
learning_rate = 0.0001
weight_decay = 0
EPOCH = 100
batch_size = 180
val_batch_size = 100

# convert data to a normalized torch.FloatTensor
train_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize((224, 224)),
                                       transforms.ToTensor(),
                                       transforms.Normalize([0.485, 0.456, 0.406],
                                                            [0.229, 0.224, 0.225])])

# choose the training and test datasets
train_data = datasets.ImageFolder(TRAIN, transform=train_transforms)
valid_data = datasets.ImageFolder(VALID, transform=valid_transforms)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(
    train_data, batch_size=batch_size, num_workers=num_workers, shuffle=True, drop_last=True)
valid_loader = torch.utils.data.DataLoader(
    valid_data, batch_size=val_batch_size,  num_workers=num_workers, shuffle=True)

images, labels = next(iter(train_loader))
images.shape, labels.shape

# val_num 筆數
val_num = len(valid_loader.dataset)

# 設定 GPU
device = torch.device("cuda")

# model = torch.load('resnet18.pt')
model = torchvision.models.resnet101(pretrained=True, progress=True)
# 提取參數fc的輸入參數
fc_features = model.fc.in_features
# 將最後輸出類別改為20
model.fc = nn.Linear(fc_features, 20)


# 遷移學習 -> frezee
for name, parameter in model.named_parameters():
    # print(name)
    if name == 'layer4.0.conv1.weight':
        break
    # if name == 'fc.weight':
    #     break
    parameter.requires_grad = False


train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    model.to(device)
# number of epochs to train the model
n_epochs = EPOCH

valid_loss_min = np.Inf  # track change in validation loss

# train_losses,valid_losses=[],[]

# 定義損失函數
# criterion = nn.MSELoss(reduction='mean')
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
# 定義優化器
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=learning_rate,
                             weight_decay=weight_decay)
# 學習率下降
scheduler = lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.9)
train_losses, val_losses = [], []
for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    print('\nrunning epoch: {}'.format(epoch))

    # train the model
    model.train()
    with tqdm(train_loader) as pbar:
        for data, target in train_loader:
            # move tensors to GPU if CUDA is available
            if train_on_gpu:
                data, target = data.to(device), target.to(device)
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            output = model(data)
            # calculate the batch loss
            # print(output)
            # print(target)
            loss = criterion(output, target)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update training loss
            train_loss += loss.item()*data.size(0)
            # 訓練條
            pbar.update(1)
            pbar.set_description('train')
            pbar.set_postfix(
                **{
                    'epochs': str('{}/{}'.format(epoch, n_epochs)),
                    'loss': loss.item(),
                    'lr': optimizer.state_dict()['param_groups'][0]['lr']
                })
        scheduler.step()

    # validate the model
    model.eval()
    val_accuracy = 0
    for data, target in tqdm(valid_loader):
        # move tensors to GPU if CUDA is available
        if train_on_gpu:
            data, target = data.to(device), target.to(device)
        # forward pass: compute predicted outputs by passing inputs to the model
        output = model(data)
        # calculate the batch loss
        loss = criterion(output, target)
        # update average validation loss
        valid_loss += loss.item()*data.size(0)
        predict_y = torch.max(output, dim=1)[1]
        val_accuracy = val_accuracy + \
            (predict_y == target.to(device)).sum().item()
    accuracy = val_accuracy / val_num

    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
    train_losses.append(train_loss)
    val_losses.append(valid_loss)
    # print training/validation statistics
    print('Training Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        train_loss, valid_loss))
    print('validation accuracy = ', accuracy)

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
            valid_loss_min,
            valid_loss))
        torch.save(model.state_dict(), 'model/model_StepLR_5.pth')
        valid_loss_min = valid_loss

# 繪製圖
plt.figure()
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.plot(train_losses, label='train_losses')
plt.plot(val_losses, label='val_losses')
plt.legend(loc='best')
plt.savefig('D:/UCL/image/losses__StepLR_5.jpg')
plt.show()
