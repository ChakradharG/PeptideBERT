import numpy as np


def load_data(task):
    tr = np.load(f'./data/{task}/train.npz')['labels']
    vl = np.load(f'./data/{task}/val.npz')['labels']
    ts = np.load(f'./data/{task}/test.npz')['labels']

    return tr, vl, ts

def label_distribution(tr, vl, ts):
    print(1 - np.count_nonzero(tr)/tr.shape[0])
    print(1 - np.count_nonzero(vl)/vl.shape[0])
    print(1 - np.count_nonzero(ts)/ts.shape[0])

