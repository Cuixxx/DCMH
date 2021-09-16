import matplotlib.pyplot as plt
from sklearn.manifold import TSNE
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from main import DCMH
import torch
from data import *
import os
def visualization(embeddings1,embeddings2, label):
    embeddings = np.vstack([embeddings1,embeddings2])

    tSNE = TSNE(n_components=2, learning_rate=100, perplexity=10)
    vectors = tSNE.fit_transform(embeddings)

    plt.scatter(vectors[:8000, 0], vectors[:8000, 1], marker='o', c=label, cmap='tab20c')
    plt.scatter(vectors[8000:, 0], vectors[8000:, 1], marker='v', c=label, cmap='tab20c')
    plt.axis('off')
    plt.colorbar()
    plt.show()

    plt.scatter(vectors[:8000, 0], vectors[:8000, 1], marker='o', c=label, cmap='tab20c')
    plt.axis('off')
    plt.colorbar()
    plt.show()
    plt.scatter(vectors[8000:, 0], vectors[8000:, 1], marker='o', c=label, cmap='tab20c')
    plt.colorbar()
    plt.axis('off')
    plt.show()

if __name__ == '__main__':
    if not (os.path.isfile('img_embeddings.npy') and os.path.isfile('txt_embeddings.npy') and os.path.isfile('labels.npy')):
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.39912226, 0.40995254, 0.37104891], [0.21165691, 0.19001945, 0.18833912])])
        train_set = RSICDset(train=True, transform=transform)

        dataloader = DataLoader(train_set, batch_size=100, shuffle=False, num_workers=5)
        model = DCMH(64)
        model = model.cuda()
        model.load_state_dict(torch.load('./models/09-10-14:17_DCMH_IR/99.pth.tar'))

        img_vectors, txt_vectors = [],[]
        labels = []
        model.eval()
        with torch.no_grad():
            for item in dataloader:
                image = item['image'].cuda()
                txt = item['txtvector'].float().cuda()
                label = item['label'].long().cuda()
                f, g = model(image, txt)
                img_vectors.append(f)
                txt_vectors.append(g)
                labels.append(label)
        img_vectors = torch.cat(img_vectors, dim=0)
        txt_vectors = torch.cat(txt_vectors, dim=0)
        labels = torch.cat(labels, dim=0)
        img_vectors = img_vectors.cpu().numpy()
        txt_vectors = txt_vectors.cpu().numpy()
        labels = labels.cpu().numpy()
        np.save('img_embeddings.npy', img_vectors)
        np.save('txt_embeddings.npy', txt_vectors)
        np.save('labels.npy', labels)

    else:
        img_vectors = np.load('img_embeddings.npy')
        txt_vectors = np.load('txt_embeddings.npy')
        labels = np.load('labels.npy')
        visualization(img_vectors, txt_vectors, labels)
        # visualization(txt_vectors, labels)
