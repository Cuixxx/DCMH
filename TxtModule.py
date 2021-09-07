import torch.nn as nn
import torch

class TxtNet(nn.Module):
    def __init__(self,len):
        super(TxtNet, self).__init__()
        self.fc1 = nn.Linear(3099, 256)
        nn.init.normal_(self.fc1.weight.data, mean=0, std=5)
        self.ReLu = nn.ReLU()
        self.fc2 = nn.Sequential(nn.Linear(256, 8192), nn.ReLU(), nn.BatchNorm1d(8192))
        self.fc3 = nn.Linear(8192, len)

    def forward(self, x):
        h = self.fc1(x)
        h = self.ReLu(h)
        h = self.fc2(h)
        h = self.fc3(h)
        return h