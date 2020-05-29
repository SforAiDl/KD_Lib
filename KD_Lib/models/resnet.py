import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(
            in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(
            planes, planes, kernel_size=3, stride=stride, padding=1, bias=False
        )
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(
            planes, self.expansion * planes, kernel_size=1, bias=False
        )
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(
                    in_planes,
                    self.expansion * planes,
                    kernel_size=1,
                    stride=stride,
                    bias=False,
                ),
                nn.BatchNorm2d(self.expansion * planes),
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, params, num_channel=3, num_classes=10):
        super(ResNet, self).__init__()
        self.in_planes = params[0]

        self.conv1 = nn.Conv2d(
            num_channel, params[0], kernel_size=3, stride=1, padding=1, bias=False
        )
        self.bn1 = nn.BatchNorm2d(params[0])
        self.layer1 = self._make_layer(block, params[1], num_blocks[0], 1)
        self.layer2 = self._make_layer(block, params[2], num_blocks[1], 2)
        self.layer3 = self._make_layer(block, params[3], num_blocks[2], 2)
        self.layer4 = self._make_layer(block, params[4], num_blocks[3], 2)
        self.linear = nn.Linear(params[4] * block.expansion, num_classes)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, out_feature=False):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        feature = out.view(out.size(0), -1)
        out = self.linear(feature)
        if not out_feature:
            return out
        else:
            return out, feature


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


class MeanResnet(ResNet):
    def __init__(self, block, num_blocks, params, num_channel=3, num_classes=10):
        super(MeanResnet, self).__init__(
            block, num_blocks, params, num_channel, num_classes
        )
        self.linear2 = nn.Linear(params[4] * block.expansion, num_classes)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out)
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        return self.linear(out), self.linear2(out)


def ResNet18(parameters, num_channel=3, num_classes=10, att=False, mean=False):
    """
    Function that creates a ResNet 18 model

    :param parameters (list or tuple): List of parameters for the model
    :param num_channel (int): Number of channels in input specimens
    :param num_classes (int): Number of classes for classification
    :param att (bool): True if attention needs to be used
    :param mean (bool): True if mean teacher model needs to be used
    """
    model = ResNet
    if att and not mean:
        model = ResnetWithAT
    elif not att and mean:
        model = MeanResnet
    return model(
        BasicBlock, [2, 2, 2, 2], parameters, num_channel, num_classes=num_classes
    )


def ResNet34(parameters, num_channel=3, num_classes=10, att=False, mean=False):
    """
    Function that creates a ResNet 34 model

    :param parameters (list or tuple): List of parameters for the model
    :param num_channel (int): Number of channels in input specimens
    :param num_classes (int): Number of classes for classification
    :param att (bool): True if attention needs to be used
    :param mean (bool): True if mean teacher model needs to be used
    """
    model = ResNet
    if att and not mean:
        model = ResnetWithAT
    elif not att and mean:
        model = MeanResnet
    return model(
        BasicBlock, [3, 4, 6, 3], parameters, num_channel, num_classes=num_classes
    )


def ResNet50(parameters, num_channel=3, num_classes=10, att=False, mean=False):
    """
    Function that creates a ResNet 50 model

    :param parameters (list or tuple): List of parameters for the model
    :param num_channel (int): Number of channels in input specimens
    :param num_classes (int): Number of classes for classification
    :param att (bool): True if attention needs to be used
    :param mean (bool): True if mean teacher model needs to be used
    """
    model = ResNet
    if att and not mean:
        model = ResnetWithAT
    elif not att and mean:
        model = MeanResnet
    return model(
        Bottleneck, [3, 4, 6, 3], parameters, num_channel, num_classes=num_classes
    )


def ResNet101(parameters, num_channel=3, num_classes=10, att=False, mean=False):
    """
    Function that creates a ResNet 101 model

    :param parameters (list or tuple): List of parameters for the model
    :param num_channel (int): Number of channels in input specimens
    :param num_classes (int): Number of classes for classification
    :param att (bool): True if attention needs to be used
    :param mean (bool): True if mean teacher model needs to be used
    """
    model = ResNet
    if att and not mean:
        model = ResnetWithAT
    elif not att and mean:
        model = MeanResnet
    return model(
        Bottleneck, [3, 4, 23, 3], parameters, num_channel, num_classes=num_classes
    )


def ResNet152(parameters, num_channel=3, num_classes=10, att=False, mean=False):
    """
    Function that creates a ResNet 152 model

    :param parameters (list or tuple): List of parameters for the model
    :param num_channel (int): Number of channels in input specimens
    :param num_classes (int): Number of classes for classification
    :param att (bool): True if attention needs to be used
    :param mean (bool): True if mean teacher model needs to be used
    """
    model = ResNet
    if att and not mean:
        model = ResnetWithAT
    elif not att and mean:
        model = MeanResnet
    return model(
        Bottleneck, [3, 8, 36, 3], parameters, num_channel, num_classes=num_classes
    )


resnet_book = {
    "18": ResNet18,
    "34": ResNet34,
    "50": ResNet50,
    "101": ResNet101,
    "152": ResNet152,
}
