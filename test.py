from data import *
from main import DCMH
import torch
import time
from tqdm import tqdm
import matplotlib.pyplot as plt

def generate_hashcode():
    hash_len = 64
    batch_size =100
    model = DCMH(hash_len,batch_size)
    model = model.cuda()
    model.load_state_dict(torch.load('./models/09-22-20:18_DCMH_IR/219.pth.tar'))#'./models/09-09-15:24_DCMH_IR/99.pth.tar'

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize([0.39912226, 0.40995254, 0.37104891], [0.21165691, 0.19001945, 0.18833912])])
    train_set = CrossModalDataset(train=True, transform=transform)
    test_set = CrossModalDataset(train=False, transform=transform)
    train_loder = DataLoader(train_set, batch_size=batch_size, shuffle=False, num_workers=5, collate_fn=collate_fn)
    test_loader = DataLoader(test_set, batch_size=batch_size, shuffle=False, num_workers=5, collate_fn=collate_fn,drop_last=True)
    model.eval()

    with torch.no_grad():
        f_buffer = []
        g_buffer = []
        label1 = []
        for item in train_loder:
            image = item[0].cuda()
            txt = item[1].float().cuda()
            label = item[2].long().cpu().numpy()
            label1.extend(label)
            length = item[4]
            f, g = model(image, txt,length)
            f_buffer.append(f)
            g_buffer.append(g)
        f_buffer = torch.cat(f_buffer, dim=0)
        g_buffer = torch.cat(g_buffer, dim=0)
        train_hashcode = torch.sign(f_buffer + g_buffer)

        f_buffer = []
        g_buffer = []
        label2 = []
        for item in test_loader:
            image = item[0].cuda()
            txt = item[1].float().cuda()
            length = item[4]
            label = item[2].long().cpu().numpy()
            label2.extend(label)
            f, g = model(image, txt, length)
            f_buffer.append(f)
            g_buffer.append(g)
        f_buffer = torch.cat(f_buffer, dim=0)
        g_buffer = torch.cat(g_buffer, dim=0)
        test_img_hash = torch.sign(f_buffer)
        test_txt_hash = torch.sign(g_buffer)
        np.save('train_hash_code', train_hashcode.cpu().numpy())
        np.save('test_img_hash', test_img_hash.cpu().numpy())
        np.save('test_txt_hash', test_txt_hash.cpu().numpy())
        return np.array(label1),np.array(label2),train_hashcode.cpu(),test_img_hash.cpu(),test_txt_hash.cpu()

def evaluate(trn_binary, trn_label, tst_binary, tst_label, K=10):
    classes = np.max(tst_label) + 1
    tst_sample_binary = tst_binary
    tst_sample_label = tst_label
    query_times = tst_sample_binary.shape[0]#10*100
    trainset_len = trn_binary.shape[0]#50000
    AP = np.zeros(query_times)#一次检索一个AP
    precision_radius = np.zeros(query_times)
    Ns = np.arange(1, trainset_len + 1)
    sum_tp = np.zeros(trainset_len)
    recall = np.zeros(trainset_len)
    total_time_start = time.time()
    with tqdm(total=query_times, desc="Query") as pbar:
        for i in range(query_times):
            query_label = tst_sample_label[i]
            query_binary = tst_sample_binary[i, :]
            query_result = np.count_nonzero(query_binary != trn_binary, axis=1) #haming distance   # don't need to divide binary length
            sort_indices = np.argsort(query_result)#np.argsort从小到大排序返回索引
            K_sort = sort_indices[0:K]#取前K个
            buffer_yes = np.equal(query_label, trn_label[sort_indices]).astype(int)
            x = np.stack((np.sort(query_result),buffer_yes),axis=0)
            n = np.sum(buffer_yes)#9400*3
            P = np.cumsum(buffer_yes) / Ns #累计求和返回数组
            # recall = np.cumsum(buffer_yes)/sum(buffer_yes)
            precision_radius[i] = P[np.where(np.sort(query_result) > 2)[0][0]-1]
            AP[i] = np.sum(P * buffer_yes) / sum(buffer_yes)
            sum_tp = sum_tp + np.cumsum(buffer_yes)
            recall = recall + np.cumsum(buffer_yes)/sum(buffer_yes)
            pbar.set_postfix({'Average Precision': '{0:1.5f}'.format(AP[i])})
            pbar.update(1)
    pbar.close()

    precision_at_k = sum_tp / Ns / query_times
    recall = recall / query_times

    plt.plot(recall, precision_at_k)
    plt.xlim(0, 1)
    plt.ylim(0, 1)
    plt.xlabel('rec')
    plt.ylabel('pre')
    plt.title('PR-curve')
    #
    plt.plot()
    plt.savefig('./fig1.png',dpi=300)
    plt.show()
    #
    index = [100, 500, 1000, 2000]
    index = [i - 1 for i in index]
    print('precision at k:', precision_at_k[index])
    print('precision within Hamming radius 2:', np.mean(precision_radius))
    map = np.mean(AP)
    print('mAP:', map)
    print('Total query time:', time.time() - total_time_start)

if __name__ == '__main__':
    label1,label2,train_hashcode,test_img_hashcode,test_txt_hashcode = generate_hashcode()
    # train_set = np.load('trainset.npy', allow_pickle=True).item()
    # train_label = train_set['labels']
    # test_set = np.load('testset.npy', allow_pickle=True).item()
    # test_label = test_set['labels']
    # train_hashcode = np.load('train_hash_code.npy', allow_pickle=True)
    # test_img_hashcode = np.load('test_img_hash.npy', allow_pickle=True)
    # test_txt_hashcode = np.load('test_txt_hash.npy', allow_pickle=True)
    # evaluate(train_hashcode, label1, test_img_hashcode, label2)
    evaluate(test_img_hashcode, label2, test_txt_hashcode, label2)
    # evaluate(train_hashcode, label1, train_hashcode, label1)
