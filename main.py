import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from data import *
from sklearn.model_selection import KFold
from OnlineMiningTripletLoss import *
import torch.optim as optim
import time
from tensorboardX import SummaryWriter
from tqdm import tqdm
import math
from ImgModule import ImgNet
from TxtModule import TxtNet, TxtNet_GRU
import torch.nn as nn


class my_tensorboarx(object):
    def __init__(self, log_dir, file_name, start_fold_time=0):
        super().__init__()
        self.writer = SummaryWriter(log_dir=log_dir)
        self.file_name = file_name
        self.epoch = 0
        self.fold_time = start_fold_time
    def draw(self, train_loss,ap_dist1,an_dist1,ap_dist2,an_dist2, val_loss,fraction1,fraction2,epoch):
        self.epoch = epoch
        self.writer.add_scalars(str(self.file_name), {
            'train_loss': train_loss,
            'ap_dist1': ap_dist1,
            'an_dist1': an_dist1,
            'ap_dist2': ap_dist2,
            'an_dist2': an_dist2,
            'val_loss': val_loss,
            'frac1': fraction1,
            'frac2': fraction2
        }, self.epoch)


    def close(self):
        self.writer.close()

class DCMH(nn.Module):
    def __init__(self,len):
        super(DCMH, self).__init__()
        self.ImageNet = ImgNet(len)
        # self.TxtNet = TxtNet(len)
        self.weight = np.load('EmbeddingWeight.npy')
        self.TxtNet = TxtNet_GRU(torch.from_numpy(self.weight), len=len)
    def forward(self, img, txt, data_length,batch_size,flag):
        f = self.ImageNet(img)
        h0 = torch.randn((2 * 2, batch_size, 1024)).cuda()
        g = self.TxtNet(txt, h0, data_length, flag)
        return f, g

def Update_hash(dataloader):

    F_buffer = []
    G_buffer = []
    # with torch.no_grad():
    for item in dataloader:
        image = item[0].cuda()
        txt = item[1].float().cuda()
        data_length = item[4]
        sequence = item[5]
        f, g = model(image, txt, data_length, batch_size,'train')

        f = f.cpu().numpy()
        g = g.cpu().numpy()

        f_list = [0]*len(item[5])
        g_list = [0] * len(item[5])
        for i,item in enumerate(sequence):
            f_list[item] = f[i, :]
            g_list[item] = g[i, :]
        f_list = np.concatenate(f_list, axis=0).reshape(-1, 64)
        g_list = np.concatenate(g_list, axis=0).reshape(-1, 64)
        f = torch.from_numpy(f_list)
        g = torch.from_numpy(g_list)
        F_buffer.append(f)
        G_buffer.append(g)

    F_buffer = torch.cat(F_buffer, dim=0)
    G_buffer = torch.cat(G_buffer, dim=0)
    B_buffer = torch.sign(F_buffer+G_buffer)

    return B_buffer.cpu()

if __name__ == '__main__':

    transform = transforms.Compose([
        transforms.ColorJitter(0.3, 0.3, 0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.39912226, 0.40995254, 0.37104891], [0.21165691, 0.19001945, 0.18833912])])

    train_set = CrossModalDataset(train=True, transform=transform)
    test_set = CrossModalDataset(train=False, transform=transform)

    kf = KFold(n_splits=10, shuffle=True, random_state=11)
    Epoch = 500
    hash_len = 64
    gamma = 2e-5
    eta = 2e-5
    batch_size = 100

    # KFold validation
    for fold, (train_idx, val_idx) in enumerate(kf.split(train_set)):
        model = DCMH(hash_len)
        model = model.cuda()
        optimizer = optim.SGD(model.parameters(), lr=2e-3, momentum=0.99, weight_decay=1e-4)

        now = time.strftime("%m-%d-%H:%M", time.localtime(time.time()))
        model_name = now + '_DCMH_IR'
        tensorboard = my_tensorboarx(log_dir='./tensorboard_data', file_name=model_name)

        Train_set, Val_set = Subset(train_set, train_idx), Subset(train_set, val_idx)

        train_loader = DataLoader(Train_set, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=5,collate_fn=collate_fn)
        hashloader = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=5,collate_fn=collate_fn)
        val_loader = DataLoader(Val_set, batch_size=batch_size, shuffle=False, drop_last=True, num_workers=5,collate_fn=collate_fn)
        fraction1, fraction2 = 1, 1


        for i in range(Epoch):
            train_loss = 0
            val_loss = 0
            model.train()
            with tqdm(total=math.ceil(len(train_loader)), desc="training") as pbar:
                for item in train_loader:
                    image = item[0].cuda()
                    txt = item[1].float().cuda()
                    label = item[2].long().cuda()
                    hash_code = item[3].cuda()
                    data_length = item[4]

                    f, g = model(image, txt, data_length, batch_size, 'train')

                    if i < 10:
                        loss1, fraction1, ap_dist1, an_dist1 = cm_batch_all_triplet_loss(labels=label, anchor=f, another=g, margin=0.2)
                        loss2, fraction2, _, _ = cm_batch_all_triplet_loss(labels=label, anchor=g, another=f, margin=0.2)
                        loss_intra = batch_all_triplet_loss(labels=label, embedings=f, margin=0.2)[0] \
                                     + batch_all_triplet_loss(labels=label, embedings=g, margin=0.2)[0]
                    else:
                        loss1, ap_dist1, an_dist1 = cm_batch_hard_triplet_loss(labels=label, anchor=f, another=g, margin=0.2)
                        loss2, _, _ = cm_batch_hard_triplet_loss(labels=label, anchor=g, another=f, margin=0.2)
                        loss_intra = batch_hard_triplet_loss(labels=label, embeddings=f, margin=0.2) \
                                     + batch_hard_triplet_loss(labels=label, embeddings=g, margin=0.2)



                    loss_q = torch.sum(torch.pow(hash_code-f, 2)+torch.pow((hash_code-g),2))
                    ones = torch.ones(batch_size, 1).cuda()
                    balance = torch.sum(torch.pow(torch.mm(f.t(), ones), 2)+torch.pow(torch.mm(g.t(), ones), 2))
                    t_loss = loss1+loss2+gamma*loss_q+eta*balance+loss_intra
                    train_loss += t_loss
                    optimizer.zero_grad()
                    t_loss.backward()
                    optimizer.step()
                    pbar.set_postfix({'loss': '{0:1.5f}'.format(t_loss)})
                    pbar.update(1)
                pbar.close()
                train_loss = train_loss / len(train_loader)
                # if i % 5 == 0:
                #     print(f[0])
            model.eval()
            with torch.no_grad():
                with tqdm(total=math.ceil(len(val_loader)), desc="validating") as pbar:
                    for item in val_loader:
                        image = item[0].cuda()
                        txt = item[1].float().cuda()
                        label = item[2].long().cuda()
                        data_length = item[4]
                        f, g = model(image, txt, data_length, batch_size, 'train')
                        hf = torch.sign(f)
                        hg = torch.sign(g)
                        # if fraction1 > 0.65:
                        if i < 10:
                            loss1, _, ap_dist2, an_dist2 = cm_batch_all_triplet_loss(labels=label, anchor=f, another=g, margin=0.2)
                            loss2, _, _, _ = cm_batch_all_triplet_loss(labels=label, anchor=g, another=f, margin=0.2)
                            loss_intra = batch_all_triplet_loss(labels=label, embedings=f, margin=0.2)[0] \
                                     + batch_all_triplet_loss(labels=label, embedings=g, margin=0.2)[0]
                        else:
                            loss1, ap_dist2, an_dist2 = cm_batch_hard_triplet_loss(labels=label, anchor=f, another=g, margin=0.2)
                            loss2, _, _ = cm_batch_hard_triplet_loss(labels=label, anchor=g, another=f, margin=0.2)
                            loss_intra = batch_hard_triplet_loss(labels=label, embeddings=f, margin=0.2) \
                                         + batch_hard_triplet_loss(labels=label, embeddings=g, margin=0.2)

                        loss_q = torch.sum(torch.pow(hf - f, 2) + torch.pow((hg - g), 2))
                        ones = torch.ones(batch_size, 1).cuda()
                        balance = torch.sum(torch.pow(torch.mm(f.t(), ones), 2) + torch.pow(torch.mm(g.t(), ones), 2))

                        v_loss = loss1 + loss2 + gamma*loss_q + eta*balance+loss_intra
                        val_loss += v_loss
                        pbar.set_postfix({'loss': '{0:1.5f}'.format(v_loss)})
                        pbar.update(1)
                    pbar.close()
                    val_loss = val_loss / len(val_loader)

                #update hash_code
                buffer = Update_hash(hashloader)
                train_set.update_buffer(buffer)

            if (i+1) % 10 == 0:
                print('saved!')
                if not os.path.isdir('./models/{}'.format(model_name)):
                    os.mkdir('./models/{}'.format(model_name))
                torch.save(model.state_dict(), './models/{}/{}.pth.tar'.format(model_name, i))
                # np.save('./models/{}/epoch{}_hashcode.npy'.format(model_name, i), buffer.numpy())

            tensorboard.draw(train_loss=train_loss, ap_dist1=ap_dist1, an_dist1=an_dist1, ap_dist2=ap_dist2, an_dist2=an_dist2, val_loss=val_loss, epoch=i, fraction1=fraction1, fraction2=fraction2)
        tensorboard.close()

