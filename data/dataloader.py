from torch.utils.data import DataLoader
import numpy as np
from data.dataset import PeptideBERTDataset


def load_data(config):
    print(f'{"="*30}{"DATA":^20}{"="*30}')

    with np.load(f'./data/{config["task"]}/train.npz') as train,\
         np.load(f'./data/{config["task"]}/val.npz') as val,\
         np.load(f'./data/{config["task"]}/test.npz') as test:
        train_inputs = train['inputs']
        train_labels = train['labels']
        val_inputs = val['inputs']
        val_labels = val['labels']
        test_inputs = test['inputs']
        test_labels = test['labels']

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
    print('Validation dataset batches: ', len(val_data_loader))
    print('Test dataset batches: ', len(test_data_loader))

    print()

    return train_data_loader, val_data_loader, test_data_loader
