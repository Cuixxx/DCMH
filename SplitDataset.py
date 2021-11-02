import json
import re
import os
import numpy as np
import random
def generate_vec(data,dict):
    vectors = [[] for _ in range(len(data))]
    files = []

    for i, item in enumerate(data):
        files.append(item['filename'])
        for sentences in item['sentences']:
            vector = []
            tokens = sentences['tokens']
            for j, token in enumerate(tokens):
                token = re.sub(r"[,.!()]", '', token).strip()
                tokens[j] = token
            for word in tokens:
                if word in dict:
                    vector.append(dict.index(word))
                else:
                    vector.append(dict.index('<unk>'))
            vectors[i].append(vector)
    return vectors, files
def gennerate_dict(train_data):
    words_list = []
    for item in train_data:
        for sentences in item['sentences']:
            tokens = sentences['tokens']
            for i, token in enumerate(tokens):
                token = re.sub(r"[,.!()]", '', token).strip()
                tokens[i] = token
            words_list += tokens
    dict = list(set(words_list))
    dict = ['<unk>'] + dict
    dict.remove('')
    dict = np.array(dict)
    # np.save('dictionary.npy', dict)
    return dict

def data_processing(data,dictionary):
        vectors,files = generate_vec(data, dictionary)

        # labels = os.listdir('/media/2T/cc/txtclasses_rsicd')
        labels = os.listdir('/media/2T/cuican/code/DCMH/DCMH/txtclasses_rsicd')
        label_list = [0]*len(files)

        for index, label in enumerate(labels):
            # with open('/media/2T/cc/txtclasses_rsicd/{}'.format(label)) as f:
            with open('/media/2T/cuican/code/DCMH/DCMH/txtclasses_rsicd/{}'.format(label)) as f:
                for line in f.readlines():
                    line = line.strip('\n')
                    label_list[files.index(line)] = index
        # print(vectors)
        files = np.array(files)
        # np.save('files.npy', files)
        vectors = np.array(vectors)
        # np.save('txtvectors.npy', vectors)
        label_list = np.array(label_list)
        # np.save('label_list.npy', label_list)
        return files, vectors, label_list


if __name__ == '__main__':
    with open('/media/2T/cuican/code/DCMH/DCMH/dataset_rsicd.json') as f:
        load_dict = json.load(f)
        index = list(range(len(load_dict['images'])))
        random.shuffle(index)
        train_index = index[:8000]
        test_index = index[8000:]
        train_data = np.array(load_dict['images'])[train_index]
    dictionary = gennerate_dict(train_data)
    files, txtvectors, label_list = data_processing(load_dict['images'],dictionary.tolist())

    train_set = {'names': files[train_index], 'txtvectors': txtvectors[train_index], 'labels': label_list[train_index]}
    test_set = {'names': files[test_index], 'txtvectors': txtvectors[test_index], 'labels': label_list[test_index] }

    # np.save('trainset.npy', train_set)
    # np.save('testset.npy', test_set)

