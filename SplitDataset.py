import os
import numpy as np
import random

if __name__ == '__main__':

    files = np.load('files.npy')
    txtvectors = np.load('txtvectors.npy')
    label_list = np.load('label_list.npy')
    index = list(range(len(files)))
    random.shuffle(index)
    train_index = index[:8000]
    test_index = index[8000:]


    train_set = {'names': files[train_index], 'txtvectors': txtvectors[train_index], 'labels': label_list[train_index]}
    test_set = {'names': files[test_index], 'txtvectors': txtvectors[test_index], 'labels': label_list[test_index]}

    np.save('trainset.npy', train_set)
    np.save('testset.npy', test_set)

