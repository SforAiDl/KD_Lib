import torch
from torchvision import datasets, transforms
import torchvision.models as models

from KD_Lib.Quantization import Dynamic_Quantizer, Static_Quantizer, QAT_Quantizer
from KD_Lib.models import ResNet18


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "mnist_data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=4,
    shuffle=True,
)

test_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "mnist_data",
        train=False,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=4,
    shuffle=True,
)


def test_dynamic_quantization():

    model_params = [4, 4, 8, 4, 4]
    model = ResNet18(model_params, 1, 10, True)
    quantizer = Dynamic_Quantizer(model, test_loader, {torch.nn.Linear})
    quantized_model = quantizer.quantize()
    quantizer.get_model_sizes()
    quantizer.get_performance_statistics()

    del model, quantizer, quantized_model


def test_static_quantization():

    model = models.quantization.resnet18(quantize=False)
    model.fc.out_features = 10
    quantizer = Static_Quantizer(model, train_loader, test_loader)
    quantized_model = quantizer.quantize(1)
    quantizer.get_model_sizes()
    quantizer.get_performance_statistics()

    del model, quantizer, quantized_model


def test_qat_quantization():

    model = models.quantization.resnet18(quantize=False)
    model.fc.out_features = 10
    optimizer = torch.optim.Adam(model.parameters())
    quantizer = QAT_Quantizer(model, train_loader, test_loader, optimizer)
    quantized_model = quantizer.quantize(1, 1, 1, 1)
    quantizer.get_model_sizes()
    quantizer.get_performance_statistics()

    del model, quantizer, quantized_model
