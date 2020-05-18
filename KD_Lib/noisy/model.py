import torch
import torch.nn as nn
import torch.nn.functional as F


class ModLeNet(nn.Module):
    def __init__(self, img_size=32, num_classes=10, in_channels=3): 
        super(ModLeNet, self).__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fc_inp = (int(self.img_size/4)**2) * 16

        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 6, 5, padding=2),         
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2),  
            nn.Conv2d(6, 16, 5, padding=2),        
            nn.Tanh(),
            nn.MaxPool2d(2, stride=2)   
        )
        self.fc = nn.Sequential(
            nn.Linear(64*16,1024),         
            nn.Tanh(),
            nn.Linear(1024,self.num_classes)            
        )
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class LeNet(nn.Module):
    def __init__(self, img_size=32, num_classes=10, in_channels=3): 
        super(LeNet, self).__init__()

        self.img_size = img_size
        self.in_channels = in_channels
        self.num_classes = num_classes
        self.fc_inp = (int((self.img_size - 12)/4)**2) * 16

        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 6, 5),         
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2),  
            nn.Conv2d(6, 16, 5),        
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2)   
        )
        self.fc = nn.Sequential(
            nn.Linear(self.fc_inp,120),         
            nn.Tanh(),
            nn.Linear(120,84),         
            nn.Tanh(),
            nn.Linear(84,self.num_classes)            
        )
        
    def forward(self, x):
        x = self.cnn(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

class NIN(nn.Module):
    def __init__(self, num_classes=10, in_channels=3):
        super(NIN, self).__init__()

        self.num_classes = num_classes
        self.in_channels = in_channels

        self.features = nn.Sequential(
            nn.Conv2d(self.in__channels, 192, 5, padding=2),
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
            nn.AvgPool2d(8, stride=1)
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

class Shallow(nn.Module):

    def __init__(self, img_size=28, hidden_size=800, num_classes=10):
        super(Shallow, self).__init__()

        self.hidden_size = hidden_size
        self.num_classes = num_classes
        self.img_size = img_size

        self.fc1 = nn.Linear(self.img_size**2, self.hidden_size)
        self.fc2 = nn.Linear(self.hidden_size, self.hidden_size)
        self.fc3 = nn.Linear(self.hidden_size, self.num_classes)

    def forward(self, x):
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = F.relu(x)
        out = self.fc3(x)

        return out
