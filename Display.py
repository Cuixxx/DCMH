import cv2
import numpy as np
import os
import torch.nn as nn
import TxtModule
import torch
from torchvision import transforms
import PIL.Image as Image
import json
import re
from main import DCMH
class TxtNet(nn.Module):
    def __init__(self,len):
        super(TxtNet, self).__init__()
        self.weight = np.load('EmbeddingWeight.npy')
        self.TxtNet = TxtModule.TxtNet_GRU(torch.from_numpy(self.weight), len=len)

    def forward(self, txt, data_length, batch_size, flag):
        h0 = torch.randn((2 * 2, batch_size, 1024)).cuda()
        g = self.TxtNet(txt, h0, data_length, flag)
        return g

class Display():
    def __init__(self,path):
        hash_len = 64
        self.batch_size = 1
        model = DCMH(hash_len)
        model = model.cuda()

        model.load_state_dict(torch.load(path))
        model.eval()
        self.textnet = model
        # self.textnet.eval()
        #####加载模型训练参数#####
        # self.textnet = TxtNet(64).cuda()
        # self.textnet.eval()
        # trained_dict = torch.load('./models/11-02-10:03_DCMH_IR/179.pth.tar')
        # net_dict = self.textnet.state_dict()
        # trained_dict = {k:v for k, v in trained_dict.items() if k in net_dict}
        # net_dict.update(trained_dict)
        # self.textnet.load_state_dict(net_dict)
        self.dict = list(np.load('dictionary.npy'))
        self.list = np.load('trainset.npy', allow_pickle=True).item()
        self.image_list = self.list['names']
        self.label_list = self.list['labels']
        self.textvectors = self.list['txtvectors']
        self.hash_list = np.load('train_hash_code.npy', allow_pickle=True)
        self.pathlist = self.image_list
        # self.gf1mul_net = Network.gf1mulNet().cuda()
        # self.gf1mul_net.load_state_dict(torch.load(path), strict=False)
        # self.gf2mul_net = Network.gf2mulNet().cuda()
        # self.gf2mul_net.load_state_dict(torch.load(path), strict=False)
        # self.gf1pan_net = Network.gf1panNet().cuda()
        # self.gf1pan_net.load_state_dict(torch.load(path), strict=False)
        # self.switch = {1: self.gf1mul_net,2: self.gf2mul_net,3: self.gf1pan_net}
        # self.hash_list = torch.load('./result/train_binary').cpu().numpy()
        # self.hash_list = np.asarray(self.hash_list, np.int32)
        # # self.hash_list = np.concatenate((self.hash_list[0], self.hash_list[1], self.hash_list[2]), axis=0)
        # self.path_list = np.load('./result/Tpath.npy')
        #
        # # self.path_list = np.concatenate((self.path_list[0], self.path_list[1], self.path_list[2]), axis=0)
        # for i,path_list in enumerate(self.path_list):
        #     for j, item in enumerate(path_list):
        #         name = item.split('.')[0].split('/')
        #         name = name[7:]
        #         name = os.path.join('/static/new/','/'.join(name))+'.jpg'
        #         self.path_list[i][j] = name

    def run(self, text):
        with torch.no_grad():
            # vector = self.textvectors[1013][0]
            vector = []
            tokens = text
            tokens = re.sub(r'[(),.!]', '', tokens).strip()
            tokens = tokens.split(' ')
            for word in tokens:
                if word in self.dict:
                    vector.append(self.dict.index(word))
                else:
                    vector.append(self.dict.index('<unk>'))
            # g = self.textnet(torch.tensor(vector).cuda(), len(vector), self.batch_size, 'test')
            _, g = self.textnet(torch.zeros([1, 3, 224, 224]).cuda(), torch.tensor(vector).unsqueeze(0).cuda(), torch.tensor([len(vector)]),1, 'train')
            hash_code = torch.sign(g)



        query_result = np.count_nonzero(hash_code.cpu().numpy() != self.hash_list, axis=1)
        # sort_indices = np.argsort(query_result)  # np.argsort从小到大排序返回索引
        # buffer_yes = np.equal(self.label_list[1013], self.label_list[sort_indices]).astype(int)
        result = self.pathlist[np.argsort(query_result)]
        result = ['/static/RSICD_images/'+item for item in result[0:500]]
        result = dict(zip(range(len(result)),result))
        j_result = json.dumps(result)
        # print(hash_code)
        return j_result
if __name__ == '__main__':
    path = './models/11-02-10:03_DCMH_IR/179.pth.tar'
    display = Display(path)
    display.run('many planes are parked next to a long building in an airport')