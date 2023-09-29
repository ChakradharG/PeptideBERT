import numpy as np
import os
from sklearn.model_selection import train_test_split
from convert_encodings import m2


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

    if not os.path.exists(f'./data/{task}'):
        os.mkdir(f'./data/{task}')

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


def combine(inputs, labels, new_inputs, new_labels):
    new_inputs = np.vstack(new_inputs)
    new_labels = np.hstack(new_labels)

    inputs = np.vstack((inputs, new_inputs))
    labels = np.hstack((labels, new_labels))

    return inputs, labels


def random_replace(inputs, labels, factor):
    new_inputs = []
    new_labels = []
    for idx in range(inputs.shape[0]):
        ip = inputs[idx]
        label = labels[idx]

        try:
            unpadded_len = np.where(ip == 0)[0][0]
        except IndexError:
            unpadded_len = len(ip)
        num_to_replace = round(unpadded_len * factor)
        indices = np.random.choice(unpadded_len, num_to_replace, replace=False)
        ip[indices] = np.random.choice(np.arange(5, 25), num_to_replace, replace=True)

        new_inputs.append(ip)
        new_labels.append(label)

    return new_inputs, new_labels


def random_delete(inputs, labels, factor):
    new_inputs = []
    new_labels = []
    for idx in range(inputs.shape[0]):
        ip = inputs[idx]
        label = labels[idx]

        try:
            unpadded_len = np.where(ip == 0)[0][0]
        except IndexError:
            unpadded_len = len(ip)
        ip = list(ip[:unpadded_len])
        num_to_delete = round(unpadded_len * factor)
        indices = np.random.choice(unpadded_len, num_to_delete, replace=False)
        for i in reversed(sorted(indices)):
            ip.pop(i)
        ip.extend([0] * (200 - len(ip)))

        new_inputs.append(np.asarray(ip))
        new_labels.append(label)

    return new_inputs, new_labels


def random_replace_with_A(inputs, labels, factor):
    new_inputs = []
    new_labels = []
    for idx in range(inputs.shape[0]):
        ip = inputs[idx]
        label = labels[idx]

        try:
            unpadded_len = np.where(ip == 0)[0][0]
        except IndexError:
            unpadded_len = len(ip)
        num_to_replace = round(unpadded_len * factor)
        indices = np.random.choice(unpadded_len, num_to_replace, replace=False)
        ip[indices] = m2['A']

        new_inputs.append(ip)
        new_labels.append(label)

    return new_inputs, new_labels


def random_swap(inputs, labels, factor):
    new_inputs = []
    new_labels = []
    for idx in range(inputs.shape[0]):
        ip = inputs[idx]
        label = labels[idx]

        try:
            unpadded_len = np.where(ip == 0)[0][0]
        except IndexError:
            unpadded_len = len(ip)
        ip = list(ip[:unpadded_len])
        num_to_swap = round(unpadded_len * factor)
        indices = np.random.choice(range(1, unpadded_len, 2), num_to_swap, replace=False)
        for i in indices:
            ip[i-1], ip[i] = ip[i], ip[i-1]
        ip.extend([0] * (200 - len(ip)))

        new_inputs.append(np.asarray(ip))
        new_labels.append(label)

    return new_inputs, new_labels


def random_insertion_with_A(inputs, labels, factor):
    new_inputs = []
    new_labels = []
    for idx in range(inputs.shape[0]):
        ip = inputs[idx]
        label = labels[idx]

        try:
            unpadded_len = np.where(ip == 0)[0][0]
        except IndexError:
            unpadded_len = len(ip)
        ip = list(ip[:unpadded_len])
        num_to_insert = round(unpadded_len * factor)
        indices = np.random.choice(unpadded_len, num_to_insert, replace=False)
        for i in indices:
            ip.insert(i, m2['A'])
        if len(ip) < 200:
            ip.extend([0] * (200 - len(ip)))
        elif len(ip) > 200:
            ip = ip[:200]

        new_inputs.append(np.asarray(ip))
        new_labels.append(label)

    return new_inputs, new_labels


def random_masking(sequences, mask_prob=0.15, mask_token_id=0):
    masked_sequences = np.copy(sequences)
    mask = np.random.rand(*sequences.shape) < mask_prob
    masked_sequences[mask] = mask_token_id
    return masked_sequences


def augment_data(task):
    with np.load(f'./data/{task}/train.npz') as train:
        inputs = train['inputs']
        labels = train['labels']

    # new_inputs1, new_labels1 = random_replace(inputs, labels, 0.02)
    # new_inputs2, new_labels2 = random_delete(inputs, labels, 0.02)
    # new_inputs3, new_labels3 = random_replace_with_A(inputs, labels, 0.02)
    new_inputs4, new_labels4 = random_swap(inputs, labels, 0.02)
    # new_inputs5, new_labels5 = random_insertion_with_A(inputs, labels, 0.02)
    #new_inputs6, new_labels6 = random_masking(inputs, mask_prob=0.15, mask_token_id=0)

    # inputs, labels = combine(inputs, labels, new_inputs1, new_labels1)
    # inputs, labels = combine(inputs, labels, new_inputs2, new_labels2)
    # inputs, labels = combine(inputs, labels, new_inputs3, new_labels3)
    inputs, labels = combine(inputs, labels, new_inputs4, new_labels4)
    # inputs, labels = combine(inputs, labels, new_inputs5, new_labels5)
    #inputs, labels = combine(inputs, labels, new_inputs6, new_labels6)

    np.savez(
        f'./data/{task}/train.npz',
        inputs=inputs,
        labels=labels
    )


def main():
    split_data('hemo')
    split_data('sol')
    split_data('nf')

    # augment_data('sol')

main()
