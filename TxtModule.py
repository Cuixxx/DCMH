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

class TxtNet_GRU(nn.Module):
    def __init__(self, weight,len):
        super(TxtNet_GRU, self).__init__()
        self.embedding = nn.Embedding.from_pretrained(weight, freeze=False)
        self.dropout = nn.Dropout(0.2)
        self.GRU = nn.GRU(300, 1024, 2, batch_first=True, bidirectional=True, dropout=0.5)
        self.fc = nn.Sequential(nn.Linear(1024, 1024), nn.ReLU(), nn.BatchNorm1d(1024))
        self.Linear = nn.Linear(1024, len)
        # self.h0 = torch.randn((2 * 2, batch_szie, 1024)).cuda()

    def forward(self, sentence,h0, sentence_length, flag):
        if flag == 'train':
            embeds = self.embedding(sentence.long())
            embeds = self.dropout(embeds)
            embeds = embeds.type(torch.float32)
            embeds = rnn.pack_padded_sequence(embeds, sentence_length, batch_first=True)
            GRU_out, _ = self.GRU(embeds, h0)
            GRU_out, len = rnn.pad_packed_sequence(GRU_out, batch_first=True)
            features = [((item[len[i]-1][:1024]+item[len[i]-1][1024:])/2.0).unsqueeze(0) for i, item in enumerate(GRU_out)]
            features = torch.cat(features, dim=0)
            features = self.fc(features)
            out = self.Linear(features)
        elif flag == 'test':
            embeds = self.embedding(sentence.long())
            embeds = self.dropout(embeds)
            embeds = embeds.type(torch.float32).unsqueeze(0)
            GRU_out, _ = self.GRU(embeds, h0)
            features = (GRU_out[:,sentence_length-1,:1024]+GRU_out[:,sentence_length-1,1024:])/2
            features = self.fc(features)
            out = self.Linear(features)

        return out