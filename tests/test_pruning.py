import torch
from torchvision import datasets, transforms

from KD_Lib.Pruning import LotteryTicketsPruner, WeightThresholdPruner
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


def test_lottery_tickets():

    teacher_params = [4, 4, 8, 4, 4]
    teacher_model = ResNet18(teacher_params, 1, 10)
    pruner = LotteryTicketsPruner(teacher_model, train_loader, test_loader)
    pruner.prune(num_iterations=2, train_epochs=1, save_models=True, prune_percent=50)

    del teacher_model, pruner


def test_weight_threshold_pruning():

    teacher_params = [4, 4, 8, 4, 4]
    teacher_model = ResNet18(teacher_params, 1, 10)
    pruner = WeightThresholdPruner(teacher_model, train_loader, test_loader)
    pruner.prune(num_iterations=2, train_epochs=1, save_models=True, threshold=0.1)
    pruner.evaluate(model_path="pruned_model_iteration_0.pt")
    pruner.get_pruning_statistics(
        model_path="pruned_model_iteration_0.pt", verbose=True
    )

    del teacher_model, pruner
