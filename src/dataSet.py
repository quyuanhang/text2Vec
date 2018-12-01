import numpy as np
import pandas as pd
from scipy import sparse
from tqdm import tqdm
from torch.utils.data import DataLoader


def load_train(filename):
    '''
    Read .rating file and Return dok matrix.
    The first line of .rating file is: n_job\t n_geek
    '''
    raw_frame = pd.read_table(filename, header=None, sep='\t')
    print(raw_frame.head())
    # raw_frame = raw_frame.iloc[:10000]
    n_job = raw_frame[0].max()
    n_geek = raw_frame[1].max()
    # Construct matrix
    mat = sparse.dok_matrix((n_job+1, n_geek+1), dtype=np.float32)
    for i, row in raw_frame.iterrows():
        user, item, rating = row
        mat[user, item] = max(rating, mat.get((user, item), 0))
    print(mat.shape)
    return mat


def load_word_emb(filename):
    words = []
    embs = []
    start = True
    with open(filename) as f:
        for line in tqdm(f):
            if start:
                start = False
                continue
            data = line.split(' ')
            word = data[0]
            emb = [float(x) for x in data[1:]]
            words.append(word)
            embs.append(emb)
    word_dict = {k: v for v, k in enumerate(words, 1)}
    dim = len(embs[0])
    embs.insert(0, [0]*dim)
    embs = np.array(embs)
    return word_dict, embs


def load_profile(filename, word_dict, position=False):
    datas = []
    with open(filename) as f:
        for line in tqdm(f):
            data = line.split(' ')
            datas.append(data)
    frame = pd.DataFrame(datas)
    if position:
        positions = frame.iloc[:, 0]
        position_dict = {k: v for v, k in enumerate(set(positions.values), 1)}
        position_dict['unk'] = 0
        frame = frame.iloc[:, 1:]
    frame = frame.applymap(lambda x: word_dict.get(x, 0))
    if position:
        return (
            position_dict,
            np.array([[position_dict.get(x, 0)] for x in positions.values]),
            frame.values)
    return frame.values


def get_train_instances(train, num_negatives, batch_size):
    n_job, n_geek = train.shape
    data = []
    # num_users = train.shape[0]
    for (u, i) in train.keys():
        # positive instance
        data.append([u, i, 1])
        # negative instances
        for _ in range(num_negatives):
            j = np.random.randint(n_geek)
            while bool(train[u, j]):
                j = np.random.randint(n_geek)
            data.append([u, j, 0])
    # import pdb
    # pdb.set_trace()
    data = np.random.permutation(data)
    data = DataLoader(data, batch_size=batch_size, shuffle=True)
    return data


def get_train_instances_multi(train, num_negatives, batch_size):
    label_dic = {
        0: [0, 0, 0],
        1: [1, 0, 0],
        2: [0, 0, 1],
        3: [1, 1, 1]
    }
    n_job, n_geek = train.shape
    data = []
    for (u, i), r in train.items():
        # positive instance
        label = label_dic[r]
        data.append([u, i, *label])
        # negative instances
        for _ in range(num_negatives):
            j = np.random.randint(0, n_geek)
            while bool(train[u, j]):
                j = np.random.randint(0, n_geek)
            data.append([u, j, 0, 0, 0])
    data = np.random.permutation(data)
    data = DataLoader(data, batch_size=batch_size, shuffle=True)
    return data


def get_train_instances_pairwise(train, batch_size):
    n_job, n_geek = train.shape
    data = []
    for (u, i), r in train.items():
        # positive instance
        j = np.random.randint(0, n_geek)
        while (u, j) in train:
            j = np.random.randint(0, n_geek)
        data.append([u, i, j])
    data = np.random.permutation(data)
    data = DataLoader(data, batch_size=batch_size, shuffle=True)
    return data


def load_test(filename, n_sample=0):
    raw_frame = pd.read_table(filename, header=None)
    if n_sample and n_sample < len(raw_frame):
        print('sampling test')
        raw_frame = raw_frame.sample(n=n_sample)
    print(raw_frame.head())
    return raw_frame.values


if __name__ == '__main__':
    word_dict, embs = load_word_emb('data/interview7.word_emb')
    geek_profile = load_profile('data/interview7.profile.geek', word_dict)
