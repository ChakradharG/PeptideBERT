import numpy as np
from sklearn.model_selection import train_test_split


def split_data(task):
    with np.load(f'./data/{task}-positive.npz') as pos,\
         np.load(f'./data/{task}-negative.npz') as neg:
        pos_data = pos['arr_0']
        neg_data = neg['arr_0']

    input_ids = np.vstack((
        pos_data,
        neg_data
    ))

    labels = np.hstack((
        np.ones(len(pos_data)),
        np.zeros(len(neg_data))
    ))

    train_val_inputs, test_inputs, train_val_labels, test_labels = train_test_split(
        input_ids, labels, test_size=0.1
    )

    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        train_val_inputs, train_val_labels, test_size=0.1
    )

    np.savez(
        f'./data/{task}/train.npz',
        inputs=train_inputs,
        labels=train_labels
    )

    np.savez(
        f'./data/{task}/val.npz',
        inputs=val_inputs,
        labels=val_labels
    )

    np.savez(
        f'./data/{task}/test.npz',
        inputs=test_inputs,
        labels=test_labels
    )


def main():
    split_data('hemo')
    split_data('sol')
    split_data('nf')


main()
