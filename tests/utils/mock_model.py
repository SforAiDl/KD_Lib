import torch
import torch.nn as nn


class MockImageClassifier(nn.Module):
    def __init__(self, size, n_classes, n_channels=3):
        super().__init__()

        if not isinstance(size, list) and not isinstance(size, tuple):
            size = (size, size)

        self.model = nn.Linear(size[0] * size[1] * n_channels, n_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        return self.model(x)
