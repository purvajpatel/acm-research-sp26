import torch
from torch.utils.data import Dataset

class SpeechDataset(Dataset):
    def __init__(self):
        # Dummy data for testing the training loop
        self.spectrograms = [torch.randn(100, 64) for _ in range(8)]
        self.phonemes = [[1, 2, 3, 4] for _ in range(8)]

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        return self.spectrograms[idx], self.phonemes[idx]

'''
FOR TRAIN.PY TO WORK, REPLACE W/
import torch
from torch.utils.data import Dataset

class SpeechDataset(Dataset):
    def __init__(self):
        # Dummy data for testing the training loop
        self.spectrograms = [torch.randn(100, 64) for _ in range(8)]
        self.phonemes = [[1, 2, 3, 4] for _ in range(8)]

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        return self.spectrograms[idx], self.phonemes[idx]



ORIGINAL

import torch
from torch.utils.data import Dataset


class SpeechDataset(Dataset):
    def __init__(self, spectrograms, phonemes):
        self.spectrograms = spectrograms
        self.phonemes = phonemes

    def __len__(self):
        return len(self.spectrograms)

    def __getitem__(self, idx):
        return self.spectrograms[idx], self.phonemes[idx]
'''