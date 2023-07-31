import torch
import numpy as np


class PeptideBERTDataset(torch.utils.data.Dataset):
    def __init__(self, input_ids, attention_masks, labels, transforms):
        self.input_ids = input_ids
        self.attention_masks = attention_masks
        self.labels = labels
        self.transforms = transforms

        self.length = len(self.input_ids)

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        input_id = self.transforms(self.input_ids[idx])
        attention_mask = self.attention_masks[idx]
        label = self.labels[idx]

        return {
            'input_ids': torch.tensor(input_id, dtype=torch.long),
            'attention_mask': torch.tensor(attention_mask, dtype=torch.long),
            'labels': torch.tensor(label, dtype=torch.float)
        }


class RandomReplace(torch.nn.Module):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, sample):
        unpadded_len = np.where(sample == 0)[0][0]
        to_replace = int(unpadded_len * self.factor)
        indices = np.random.choice(unpadded_len, to_replace, replace=False)
        sample[indices] = np.random.choice(np.arange(5, 25), to_replace, replace=True)

        return sample
