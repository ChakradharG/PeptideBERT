import numpy as np

m1 = [
    '[PAD]','A','R','N','D','C','Q','E','G','H',
    'I','L','K','M','F','P','S','T','W','Y','V'
]
m2 = dict(zip(
    ['[PAD]','[UNK]','[CLS]','[SEP]','[MASK]','L',
    'A','G','V','E','S','I','K','R','D','T','P','N',
    'Q','F','Y','M','H','C','W','X','U','B','Z','O'],
    range(30)
))


def func1(file):
    f = np.load(file)
    arr = np.array(
        list(map(
            lambda x: m2[m1[int(x)]],
            f['arr_0'].flat
        ))
    ).reshape(f['arr_0'].shape)

    np.savez(
        file,
        arr_0=arr
    )


def func2(task):
    func1(f'./data/{task}-positive.npz')
    func1(f'./data/{task}-negative.npz')


def main():
    func2('hemo')
    func2('sol')
    func2('nf')
