import torch.nn as nn


class ModLeNet(nn.Module):
    """
    Implementation of a ModLeNet model

    :param img_size (int): Dimension of input image
    :param hidden_size (int): Hidden layer dimension
    :param num_classes (int): Number of classes for classification
    :param in_channels (int): Number of channels in input specimens
    """

    def __init__(self, img_size=32, num_classes=10, in_channels=3):
        super(ModLeNet, self).__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fc_inp = (int(self.img_size / 4) ** 2) * 16

        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 6, 5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5, padding=2),
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(64 * 16, 1024), nn.Tanh(), nn.Linear(1024, self.num_classes)
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


class LeNet(nn.Module):
    """
    Implementation of a LeNet model

    :param img_size (int): Dimension of input image
    :param hidden_size (int): Hidden layer dimension
    :param num_classes (int): Number of classes for classification
    :param in_channels (int): Number of channels in input specimens
    """

    def __init__(self, img_size=32, num_classes=10, in_channels=3):
        super(LeNet, self).__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fc_inp = (int((self.img_size - 12) / 4) ** 2) * 16

        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 6, 5),
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2),
            nn.Conv2d(6, 16, 5),
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2),
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fc_inp, 120),
            nn.Tanh(),
            nn.Linear(120, 84),
            nn.Tanh(),
            nn.Linear(84, self.num_classes),
        )

    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
