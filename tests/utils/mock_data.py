import torch
from torch.utils.data import Dataset


class MockVisionDataset(Dataset):
    def __init__(self, size, n_classes, length, n_channels):

        self.length = length

        if not isinstance(size, list) and not isinstance(size, tuple):
            size = (size, size)

        self.imgs = torch.randn(length, n_channels, *size)
        self.labels = torch.randint(0, n_classes, (length,))

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        return self.imgs[idx], self.labels[idx]
