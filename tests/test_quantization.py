from copy import deepcopy

import torch
from torch.utils.data import DataLoader

from KD_Lib.Quantization import Dynamic_Quantizer, QAT_Quantizer, Static_Quantizer

from .utils import MockImageClassifier, MockVisionDataset

img_size = (32, 32)
img_channels = 3
n_classes = 10
len_dataset = 4
batch_size = 2

train_loader = test_loader = DataLoader(
    MockVisionDataset(
        size=img_size, n_classes=n_classes, length=len_dataset, n_channels=img_channels
    ),
    batch_size=batch_size,
)

mock_model = MockImageClassifier(
    size=img_size, n_classes=n_classes, n_channels=img_channels
)


def test_dynamic_quantization():

    model = deepcopy(mock_model)

    quantizer = Dynamic_Quantizer(model, test_loader, {torch.nn.Linear})
    _ = quantizer.quantize()
    quantizer.get_model_sizes()
    quantizer.get_performance_statistics()


def test_static_quantization():

    model = deepcopy(mock_model)

    quantizer = Static_Quantizer(model, train_loader, test_loader)
    _ = quantizer.quantize(1)
    quantizer.get_model_sizes()
    quantizer.get_performance_statistics()


def test_qat_quantization():

    model = deepcopy(mock_model)

    optimizer = torch.optim.Adam(model.parameters())
    quantizer = QAT_Quantizer(model, train_loader, test_loader, optimizer)
    _ = quantizer.quantize(1, 1, 1, 1)
    quantizer.get_model_sizes()
    quantizer.get_performance_statistics()
