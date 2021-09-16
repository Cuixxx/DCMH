import torch
import numpy as np
import json
import re
import pickle
from tqdm import tqdm
import math

def GenerateDicts():
    words = ['PAD']
    embeds = np.zeros(shape=[1,300],dtype=np.float32)

    with open('/media/2T/cc/glove.6B.300d.txt', 'r', encoding='utf-8') as f:
        lines = f.readlines()
        with tqdm(total=math.ceil(len(lines)), desc="processing") as pbar:
            for line in lines:
                values = line.split()
                words.append(values[0])
                vector = np.asarray(values[1:], 'float32')
                embeds = np.concatenate((embeds, vector.reshape(1,300)),axis=0)
            # pbar.set_postfix({'loss': '{0:1.5f}'.format(v_loss)})
                pbar.update(1)
            pbar.close()
    w2idx_dict = dict(zip(words, range(len(words))))

    idx2w_dict = dict(zip(range(len(words)), words))
    assert embeds.shape[0] == len(words)
    glove_data ={'w2idx_dict':w2idx_dict, 'idx2w_dict':idx2w_dict, 'embed_matrix':embeds}
    pickle.dump(glove_data, open('glove_data_6B_300d.pkl','wb'))
    # file = open('glove_data_6B_300d.json','w')
    # file.write(glove_data)
    # file.close()

def GenerateWeights(vocab,emb_dim):
    matrix_len = len(vocab)
    weights = np.zeros([matrix_len,emb_dim])
    glove_data = pickle.load(open('glove_data_6B_300d.pkl', 'rb'))
    w2idx_dict = glove_data['w2idx_dict']
    embeddings = glove_data['embed_matrix']
    unk_count=0
    for idx, word in enumerate(vocab):
        try:
            weights[idx] = embeddings[w2idx_dict[word]]
        except KeyError:
            weights[idx] = np.random.normal(size=(emb_dim,))
            unk_count += 1
    print(unk_count)
    np.save('EmbeddingWeight.npy', weights)
    return weights

if __name__ == '__main__':
    # GenerateDicts()
    dictionary = np.load('dictionary.npy')
    GenerateWeights(dictionary,emb_dim=300)
    # dicts = pickle.load(open('glove_data_6B_300d.pkl','rb'))
    # print(dicts)

