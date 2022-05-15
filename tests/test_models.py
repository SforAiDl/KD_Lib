import torch

from KD_Lib.models import (
    LeNet,
    LSTMNet,
    ModLeNet,
    NetworkInNetwork,
    ResNet18,
    ResNet34,
    ResNet50,
    ResNet101,
    ResNet152,
    Shallow,
)

sample_input = torch.randn(size=(1, 3, 32, 32), requires_grad=False)


def test_resnet():

    params = [4, 4, 8, 8, 16]

    model = ResNet18(params)
    _ = model(sample_input)

    model = ResNet34(params)
    _ = model(sample_input)

    model = ResNet50(params)
    _ = model(sample_input)

    model = ResNet101(params)
    _ = model(sample_input)

    model = ResNet152(params)
    _ = model(sample_input)

    model = ResNet34(params, att=True)
    _ = model(sample_input)

    model = ResNet34(params, mean=True)
    _ = model(sample_input)

    model = ResNet101(params, att=True)
    _ = model(sample_input)

    model = ResNet101(params, mean=True)
    _ = model(sample_input)


def test_attention_model():

    params = [4, 4, 8, 8, 16]
    model = ResNet152(params, att=True)
    _ = model(sample_input)


def test_NIN():

    model = NetworkInNetwork(10, 3)
    _ = model(sample_input)


def test_shallow():

    model = Shallow(32, num_channels=3)
    _ = model(sample_input)


def test_lenet():

    model = LeNet()
    _ = model(sample_input)


def test_modlenet():

    model = ModLeNet()
    _ = model(sample_input)


def test_LSTMNet():

    sample_input = torch.tensor([[1, 2, 8, 3, 2], [2, 4, 99, 1, 7]])
    sample_lengths = torch.tensor([5, 5])

    # Simple LSTM
    model = LSTMNet(num_classes=2, dropout_prob=0.5)
    _ = model(sample_input, sample_lengths)

    # Bidirectional LSTM
    model = LSTMNet(num_classes=2, dropout_prob=0.5, bidirectional=True)
    _ = model(sample_input, sample_lengths)
