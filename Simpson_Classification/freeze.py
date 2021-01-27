import torch
import torchvision
from torchsummary import summary
import torch.nn as nn

device = torch.device("cuda")
model = torchvision.models.resnet101(pretrained=True, progress=True)

# 顯示可用涵式
print(dir(model))

# 提取參數fc的輸入參數
fc_features = model.fc.in_features
# 將最後輸出類別改為20
model.fc = nn.Linear(fc_features, 20)

for name, parameter in model.named_parameters():
    print(name)
    if name == 'layer4.0.conv1.weight':
        break
    parameter.requires_grad = False

# 輸出模型參數
summary(model.to(device),(3,224,224))