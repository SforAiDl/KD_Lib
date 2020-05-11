import torch
import torch.nn as nn
import torch.nn.functional as F


class large(nn.Module):
    def __init__(self):
        super(large, self).__init__()

        self.fc1 = nn.Linear(28*28, 1200)
        self.fc2 = nn.Linear(1200, 1200)
        self.fc3 = nn.Linear(1200, 10)
        self.dropout = nn.Dropout2d(0.5)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = F.relu(x)
        x = self.dropout(x)
        out = self.fc3(x)

        return out


class distil(nn.Module):

    def __init__(self):
        super(distil, self).__init__()

        self.fc1 = nn.Linear(28*28, 800)
        self.fc2 = nn.Linear(800, 800)
        self.fc3 = nn.Linear(800, 10)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        out = self.fc3(x)

        return out
