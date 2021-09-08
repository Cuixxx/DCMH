import json
import re
import numpy as np
import os
import random
from torch.utils.data import Dataset,DataLoader,random_split,Subset
import PIL.Image as Image
import torchvision.transforms as transforms
import torch
##ceshi
#1123
def data_processing():
    with open('/media/2T/cc/RSICD/dataset_rsicd.json') as f:
        load_dict = json.load(f)
        words_list = []
        for item in load_dict['images']:
            for sentences in item['sentences']:
                # tokens = sentences['tokens']
                tokens = sentences['raw']
                tokens = re.sub(r'[(),.!]','',tokens).strip()
                tokens = tokens.split(' ')
                words_list += tokens
        dict = list(set(words_list))
        dict.remove('')

        # hist = [0]*len(dict)
        # for word in words_list:
        #     if word in dict:
        #         hist[dict.index(word)] += 1
        vectors = [[] for i in range(len(load_dict['images']))]
        files = []
        for i,item in enumerate(load_dict['images']):
            files.append(item['filename'])
            for sentences in item['sentences']:
                # tokens = sentences['tokens']
                vector = [0]*len(dict)
                tokens = sentences['raw']
                tokens = re.sub(r'[(),.!]','',tokens).strip()
                tokens = tokens.split(' ')
                for word in tokens:
                    if word in dict:
                        vector[dict.index(word)] = 1
                # print(vector)
                vectors[i].append(vector)

        labels = os.listdir('/media/2T/cc/txtclasses_rsicd')
        label_list = [0]*len(files)

        for index,label in enumerate(labels):
            with open('/media/2T/cc/txtclasses_rsicd/{}'.format(label)) as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    label_list[files.index(line)]=index

        files = np.array(files)
        np.save('files.npy',files)
        vectors = np.array(vectors)
        np.save('txtvectors.npy',vectors)
        dict = np.array(dict)
        np.save('dictionary.npy',dict)
        label_list = np.array(label_list)
        np.save('label_list.npy',label_list)
        # print(load_dict)

class RSICDset(Dataset):
    def __init__(self,train=True,hash_len=64, transform=None):
        if train == True:
            self.list = np.load('trainset.npy', allow_pickle=True).item()
        else:
            self.list = np.load('testset.npy', allow_pickle=True).item()

        self.label_list = self.list['labels']
        self.txtvectors = self.list['txtvectors']
        self.image_list = self.list['names']

        self.transform = transform
        self.B_buffer = torch.randn(len(self.label_list),hash_len)

    def __getitem__(self,index):
        file_name = self.image_list[index]
        image = Image.open('/media/2T/cc/RSICD/RSICD_images/'+file_name)
        if self.transform:
            image = self.transform(image)

        txtvector = self.txtvectors[index][random.randint(0, 4)]
        label = self.label_list[index]
        hash_code = self.B_buffer[index]
        result = {'image': image, 'txtvector': txtvector, 'label': label, 'hash_code':hash_code}
        return result

    def __len__(self):
        return len(self.txtvectors)

    def update_buffer(self, buffer):
        self.B_buffer = buffer


if __name__ == '__main__':
    transform = transforms.ToTensor()
    dataset = RSICDset(transform)


    dataloader = DataLoader(dataset=dataset,batch_size=30,shuffle=True,num_workers=10)
    for item in dataloader:
        image = item['image']
        vector = item['txtvector']
        label = item['label']
    # data_processing()

    # label_list = np.load('label_list.npy',allow_pickle=True)
    # txtvectors = np.load('txtvectors.npy')
    # for c in label_list:
    #     labels = random.sample(c,10)
    #     vctors = [txtvectors[item[1]] for item in labels]