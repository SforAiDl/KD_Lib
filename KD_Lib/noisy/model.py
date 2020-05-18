import torch.nn as nn
import torch.nn.functional as F


class LeNet(nn.Module):
    def __init__(self, num_classes=10, in_channels=3): 
        super(LeNet, self).__init__()

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.cnn = nn.Sequential(
            nn.Conv2d(self.in_channels, 6, 5),         
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2),  
            nn.Conv2d(6, 16, 5),        
            nn.Tanh(),
            nn.AvgPool2d(2, stride=2)   
        )
        self.fc = nn.Sequential(
            nn.Linear(400,120),         
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

        self.in_channels = in_channels
        self.num_classes = num_classes

        self.classifier = nn.Sequential(
                nn.Conv2d(self.num_classes, 192, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 160, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(160,  96, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(96, 192, kernel_size=5, stride=1, padding=2),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=3, stride=2, padding=1),
                nn.Dropout(0.5),

                nn.Conv2d(192, 192, kernel_size=3, stride=1, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(192, 192, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.Conv2d(192,  10, kernel_size=1, stride=1, padding=0),
                nn.ReLU(inplace=True),
                nn.AvgPool2d(kernel_size=8, stride=1, padding=0),

                )

    def forward(self, x):
        x = self.classifier(x)
        x = x.view(x.size(0), self.num_classes)
        return x

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
