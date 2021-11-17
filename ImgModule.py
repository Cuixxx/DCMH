import torchvision.models as models
import torch.nn as nn

class ImgNet(nn.Module):
    def __init__(self,len):
        super(ImgNet, self).__init__()
        feats = list(models.resnet18(pretrained=True).children())
        print(feats)
        self.feature = nn.Sequential(*feats[0:9])
        self.hash = nn.Linear(512, len)
    def forward(self, x) :
        h = self.feature(x)
        h = h.reshape([h.shape[0],-1])
        h = self.hash(h)
        return h

if __name__ == '__main__':
    net = ImgNet(10)