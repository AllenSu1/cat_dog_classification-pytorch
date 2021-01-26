import torch
from pathlib import Path
from torchvision import datasets ,models,transforms
import numpy as np
import matplotlib.pyplot as plt
import torch
from tqdm import tqdm
# from model import CNN_Model 
from ResNet18 import resnet18
import numpy as np
from torch.utils.data.sampler import SubsetRandomSampler
import torch.nn as nn
from torch.optim import lr_scheduler
import torchvision


train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('CUDA is not available.  Training on CPU ...')
else:
    print('CUDA is available!  Training on GPU ...')

PATH_train="D:\\Dataset\\dogs_and_cats\\z_test\\train"
PATH_val="D:\\Dataset\\dogs_and_cats\\z_test\\val"
PATH_test="D:\\Dataset\\dogs_and_cats\\z_test\\test_1"

TRAIN =Path(PATH_train)
VALID = Path(PATH_val)
TEST=Path(PATH_test)
print(TRAIN)
print(VALID)
print(TEST)


# number of subprocesses to use for data loading
num_workers = 0
#設計超參數
learning_rate = 0.00001
weight_decay = 0
EPOCH = 10
batch_size = 16
val_batch_size = 8

# convert data to a normalized torch.FloatTensor

train_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),        
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

valid_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

test_transforms = transforms.Compose([transforms.Resize((224,224)),
                                      transforms.ToTensor(),
                                      transforms.Normalize([0.485, 0.456, 0.406],
                                                           [0.229, 0.224, 0.225])])

# choose the training and test datasets
train_data = datasets.ImageFolder(TRAIN, transform=train_transforms)
valid_data = datasets.ImageFolder(VALID,transform=valid_transforms)
test_data = datasets.ImageFolder(TEST, transform=test_transforms)

print(train_data.class_to_idx)
print(valid_data.class_to_idx)

# prepare data loaders (combine dataset and sampler)
train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size, num_workers=num_workers,shuffle=True)
valid_loader = torch.utils.data.DataLoader(valid_data, batch_size=val_batch_size,  num_workers=num_workers,shuffle=True)
test_loader = torch.utils.data.DataLoader(test_data, batch_size=batch_size,  num_workers=num_workers)

images,labels=next(iter(train_loader))
images.shape,labels.shape

import matplotlib.pyplot as plt
# %matplotlib inline
classes = ['cat','dog']
mean , std = torch.tensor([0.485, 0.456, 0.406]),torch.tensor([0.229, 0.224, 0.225])

def denormalize(image):
  image = transforms.Normalize(-mean/std,1/std)(image) #denormalize
  image = image.permute(1,2,0) #Changing from 3x224x224 to 224x224x3
  image = torch.clamp(image,0,1)
  return image

# helper function to un-normalize and display an image
def imshow(img):
    img = denormalize(img) 
    plt.imshow(img)

# obtain one batch of training images
dataiter = iter(train_loader)
images, labels = dataiter.next()
 # convert images to numpy for display

val_num = 1000

# 設定 GPU
device = torch.device("cuda")

# 匯入模型
# model = CNN_Model()
# model = torch.load('resnet18.pt')
model = torchvision.models.resnet101(pretrained=True, progress=True)
train_on_gpu = torch.cuda.is_available()
if train_on_gpu:
    model.to(device)
#number of epochs to train the model
n_epochs = EPOCH

valid_loss_min = np.Inf # track change in validation loss

#train_losses,valid_losses=[],[]

# 定義損失函數
# criterion = nn.MSELoss(reduction='mean')
criterion = nn.CrossEntropyLoss()
criterion = criterion.to(device)
# 定義優化器
optimizer = torch.optim.Adam(params=model.parameters(),
                             lr=learning_rate,
                             weight_decay=weight_decay)
scheduler = lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.9)

for epoch in range(1, n_epochs+1):

    # keep track of training and validation loss
    train_loss = 0.0
    valid_loss = 0.0
    print('\nrunning epoch: {}'.format(epoch))
    ###################
    # train the model #
    ###################
    model.train()
    for data, target in tqdm(train_loader):
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
        
    ######################    
    # validate the model #
    ######################
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
        val_accuracy = val_accuracy + (predict_y == target.to(device)).sum().item()
    accuracy = val_accuracy / val_num               
   
    # calculate average losses
    #train_losses.append(train_loss/len(train_loader.dataset))
    #valid_losses.append(valid_loss.item()/len(valid_loader.dataset)
    train_loss = train_loss/len(train_loader.dataset)
    valid_loss = valid_loss/len(valid_loader.dataset)
    # print training/validation statistics 
    print('Training Loss: {:.6f} \tValidation Loss: {:.6f}'.format(
        train_loss, valid_loss))
    print('validation accuracy = ', accuracy)  

    # save model if validation loss has decreased
    if valid_loss <= valid_loss_min:
        print('Validation loss decreased ({:.6f} --> {:.6f}).  Saving model ...'.format(
        valid_loss_min,
        valid_loss))
        torch.save(model.state_dict(), 'model/model.pth')
        valid_loss_min = valid_loss
