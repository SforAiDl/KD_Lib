from KD_Lib.utils import Pipeline
from KD_Lib.KD import VanillaKD
from KD_Lib.Pruning import Lottery_Tickets_Pruner
from KD_Lib.Quantization import Dynamic_Quantizer
from KD_Lib.models import Shallow

import torch
from torchvision import datasets, transforms
import torch.optim as optim


train_loader = torch.utils.data.DataLoader(
    datasets.MNIST(
        "mnist_data",
        train=True,
        download=True,
        transform=transforms.Compose(
            [transforms.ToTensor(), transforms.Normalize((0.1307,), (0.3081,))]
        ),
    ),
    batch_size=32,
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
    batch_size=32,
    shuffle=True,
)


def test_Pipeline():
    teacher = Shallow(hidden_size=400)
    student = Shallow(hidden_size=100)

    t_optimizer = optim.SGD(teac.parameters(), 0.01)
    s_optimizer = optim.SGD(stud.parameters(), 0.01)

    distiller = VanillaKD(
        teacher, student, train_loader, test_loader, t_optimizer, s_optimizer
    )

    pruner = Lottery_Tickets_Pruner(student, train_loader, test_loader)

    quantizer = Dynamic_Quantizer(student, test_loader, {torch.nn.Linear})

    pipe = Pipeline([distiller, pruner, quantizer], 1)
    pipe.train()
