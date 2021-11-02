
import numpy as np
import os
import random
from torch.utils.data import Dataset,DataLoader,random_split,Subset
import PIL.Image as Image
import torchvision.transforms as transforms
import torch
import torch.nn.utils.rnn as rnn
import TxtModule


        # print(load_dict)

class RSICDset(Dataset):
    def __init__(self, train=True, hash_len=64, transform=None):
        if train == True:
            self.list = np.load('trainset.npy', allow_pickle=True).item()
        else:
            self.list = np.load('testset.npy', allow_pickle=True).item()

        self.label_list = self.list['labels']
        self.txtvectors = self.list['txtvectors']
        self.image_list = self.list['names']

        self.transform = transform
        self.B_buffer = torch.sign(torch.randn(len(self.label_list), hash_len))
        # self.index_list = []

    def __getitem__(self, index):
        file_name = self.image_list[index]
        # image = Image.open('/media/2T/cc/RSICD/RSICD_images/'+file_name)
        image = Image.open('/media/2T/cuican/code/DCMH/DCMH/static/RSICD_images/' + file_name)
        if self.transform:
            image = self.transform(image)

        txtvector = self.txtvectors[index][random.randint(0, 4)]
        label = self.label_list[index]
        hash_code = self.B_buffer[index]
        result = {'image': image, 'txtvector': txtvector, 'label': label, 'hash_code':hash_code}
        # self.index_list.append(index)
        return result

    def __len__(self):
        return len(self.txtvectors)

    def update_buffer(self, buffer):
        self.B_buffer = buffer

class CrossModalDataset(Dataset):
    def __init__(self, train=True, hash_len=64, transform=None):
        if train == True:
            self.list = np.load('trainset.npy', allow_pickle=True).item()
        else:
            self.list = np.load('testset.npy', allow_pickle=True).item()

        self.label_list = self.list['labels']
        self.txtvectors = self.list['txtvectors']
        self.image_list = self.list['names']

        self.transform = transform
        self.B_buffer = torch.sign(torch.randn(len(self.label_list), hash_len))
        # self.index_list = []

    def __getitem__(self, index):
        file_name = self.image_list[index]
        # image = Image.open('/media/2T/cc/RSICD/RSICD_images/'+file_name)
        image = Image.open('/media/2T/cuican/code/DCMH/DCMH/static/RSICD_images/' + file_name)
        if self.transform:
            image = self.transform(image)

        txtvector = self.txtvectors[index][random.randint(0, 4)]
        while len(txtvector) <= 0:
            txtvector = self.txtvectors[index][random.randint(0, 4)]
        label = self.label_list[index]
        hash_code = self.B_buffer[index]

        return image, txtvector, label, hash_code

    def __len__(self):
        return len(self.txtvectors)

    def update_buffer(self, buffer):
        self.B_buffer = buffer

def collate_fn(data):
    length = [len(i[1]) for i in data]
    sequence = sorted(range(len(length)), key=lambda k: length[k], reverse=True)
    data.sort(key=lambda x: len(x[1]), reverse=True)
    data_length = [len(sq[1]) for sq in data]
    img = [i[0].unsqueeze(0) for i in data]
    img = torch.cat(img, dim=0)
    x = [torch.tensor(i[1]) for i in data]
    label = [i[2] for i in data]
    hash_code = [i[3].unsqueeze(0) for i in data]
    hash_code = torch.cat(hash_code, dim=0)

    txtvector = rnn.pad_sequence(x, batch_first=True, padding_value=0)
    return img, txtvector, torch.tensor(label, dtype=torch.float32), hash_code, data_length, sequence

if __name__ == '__main__':
    weight = np.load('EmbeddingWeight.npy')
    model = TxtModule.TxtNet_GRU(weight=torch.from_numpy(weight), batch_szie=5)
    transform = transforms.ToTensor()
    dataset = CrossModalDataset(train=True, transform=transform)
    dataloader = DataLoader(dataset=dataset, batch_size=5, shuffle=True, num_workers=10, collate_fn=collate_fn)
    for item in dataloader:
        image = item[0]
        vector = item[1]
        label = item[2]
        hash_code = item[3]
        data_length = item[4]
        model(vector, data_length)
        print(data_length)
    # data_processing()