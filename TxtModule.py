import torch.nn as nn
import torch
# from torchvision.model
import torch.nn.utils.rnn as rnn

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

class TxtNet_lstm(nn.Module):
    def __init__(self,weight):
        super(TxtNet_lstm, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(weight,freeze=False)
        self.embedding_dropout = nn.Dropout(0.2)
        self.GRU = nn.GRU(300, 256, 3, batch_first=True,bidirectional=True)
        self.Linear = nn.Linear(256, 64)




    def forward(self,sentence,sentence_length):
        embeds = self.embedding(sentence.long())
        embeds = rnn.pack_padded_sequence(embeds,sentence_length,batch_first=True)
        h0 = torch.randn((3*2,len(sentence),256))
        lstm_out,_ = self.GRU(embeds,h0)