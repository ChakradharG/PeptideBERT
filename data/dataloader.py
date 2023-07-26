from torch.utils.data import DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
from data.dataset import PeptideBERTDataset


def load_data(config):
    print(f'{"="*30}{"DATA":^20}{"="*30}')

    with np.load(f'./data/{config["task"]}-positive.npz') as pos,\
         np.load(f'./data/{config["task"]}-negative.npz') as neg:
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

    config['vocab_size'] = input_ids.max() + 1

    train_val_inputs, test_inputs, train_val_labels, test_labels = train_test_split(
        input_ids, labels, test_size=0.1, random_state=config['random_state']
    )

    train_inputs, val_inputs, train_labels, val_labels = train_test_split(
        train_val_inputs, train_val_labels, test_size=0.09/0.81, random_state=config['random_state']
    )

    attention_mask = np.asarray(train_inputs > 0, dtype=np.float64)
    attention_mask_val = np.asarray(val_inputs > 0, dtype=np.float64)
    attention_mask_test = np.asarray(test_inputs > 0, dtype=np.float64)

    train_dataset = PeptideBERTDataset(input_ids=train_inputs, attention_masks=attention_mask, labels=train_labels)
    val_dataset = PeptideBERTDataset(input_ids=val_inputs, attention_masks=attention_mask_val, labels=val_labels)
    test_dataset = PeptideBERTDataset(input_ids=test_inputs, attention_masks=attention_mask_test, labels=test_labels)

    train_data_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True
    )

    val_data_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    test_data_loader = DataLoader(
        test_dataset,
        batch_size=config['batch_size'],
        shuffle=False
    )

    print('Batch size: ', config['batch_size'])

    print('Train dataset samples: ', len(train_dataset))
    print('Validation dataset samples: ', len(val_dataset))
    print('Test dataset samples: ', len(test_dataset))

    print('Train dataset batches: ', len(train_data_loader))
    print('Validataion dataset batches: ', len(val_data_loader))
    print('Test dataset batches: ', len(test_data_loader))

    print()

    return train_data_loader, val_data_loader, test_data_loader
