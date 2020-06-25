from torch import nn


class NetworkInNetwork(nn.Module):
    """
    Implementation of a Network In Network model

    :param num_classes (int): Number of classes for classification
    :param in_channels (int): Number of channels in input specimens
    """

    def __init__(self, num_classes=10, in_channels=3):
        super(NetworkInNetwork, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        self.features = nn.Sequential(
            nn.Conv2d(self.in_channels, 192, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 160, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(160, 96, 1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, ceil_mode=True),
            nn.Dropout(inplace=True),
            nn.Conv2d(96, 192, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(3, stride=2, ceil_mode=True),
            nn.Dropout(inplace=True),
            nn.Conv2d(192, 192, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, 192, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(192, self.num_classes, 1),
            nn.ReLU(inplace=True),
            nn.AvgPool2d(8, stride=1),
        )
        self._initialize_weights()

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), self.num_classes)
        return x

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                m.weight.data.normal_(0, 0.05)
                if m.bias is not None:
                    m.bias.data.zero_()
