import torch.nn.functional as F

from KD_Lib.models.resnet import ResNet


class ResnetWithAT(ResNet):
    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        at1 = self.layer1(out)
        at2 = self.layer2(at1)
        at3 = self.layer3(at2)
        at4 = self.layer4(at3)
        out = F.avg_pool2d(at4, 4)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        return out, at1, at2, at3, at4
