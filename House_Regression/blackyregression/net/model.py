# requirement

# matplotlib==3.2.2
# pandas==1.0.5
# torch==1.7.0
# numpy==1.17.0
# ipython==7.19.0
# tensorboard
# tensorboardX

# 模型
import torch
import torch.nn as nn
import torch.nn.functional as F
from tensorboardX import SummaryWriter


def init_weights(m):
    if type(m) == nn.Linear:
        torch.nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


class Net(nn.Module):
    def __init__(self, features):
        super(Net, self).__init__()

        self.layer1 = nn.Sequential(nn.Linear(features, 128), nn.ReLU(True))
        self.layer2 = nn.Sequential(nn.Linear(128, 256), nn.ReLU(True))
        self.layer3 = nn.Sequential(nn.Linear(256, 512), nn.ReLU(True))
        self.layer4 = nn.Sequential(nn.Linear(512, 1024), nn.ReLU(True))
        self.layer5 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(True))
        self.layer6 = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(True))
        self.layer7 = nn.Sequential(nn.Linear(1024, 512), nn.ReLU(True))
        self.layer8 = nn.Sequential(nn.Linear(512, 256), nn.ReLU(True))
        self.layer9 = nn.Sequential(nn.Linear(256, 128), nn.ReLU(True))
        self.layer10 = nn.Sequential(nn.Linear(128, 64), nn.ReLU(True))
        self.layer11 = nn.Sequential(nn.Linear(64, 32), nn.ReLU(True))
        self.layer12 = nn.Sequential(nn.Linear(32, 16), nn.ReLU(True))
        self.layer13 = nn.Sequential(nn.Linear(16, 1))

        self.apply(init_weights)  # 初始化權重

    def forward(self, x):

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.layer5(x)
        x = self.layer6(x)
        x = self.layer7(x)
        x = self.layer8(x)
        x = self.layer9(x)
        x = self.layer10(x)
        x = self.layer11(x)
        x = self.layer12(x)
        x = self.layer13(x)
        return x


x = torch.rand(1, 20)
model = Net(20)

with SummaryWriter(comment='Net') as w:
    w.add_graph(model, x)
