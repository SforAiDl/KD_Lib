from copy import deepcopy

from torch.utils.data import DataLoader

from KD_Lib.Pruning import LotteryTicketsPruner, WeightThresholdPruner

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


def test_lottery_tickets():

    model = deepcopy(mock_model)

    pruner = LotteryTicketsPruner(model, train_loader, test_loader)
    pruner.prune(num_iterations=2, train_epochs=1, save_models=True, prune_percent=50)


def test_weight_threshold_pruning():

    model = deepcopy(mock_model)

    pruner = WeightThresholdPruner(model, train_loader, test_loader)
    pruner.prune(num_iterations=2, train_epochs=1, save_models=True, threshold=0.1)
    pruner.get_pruning_statistics(
        model_path="pruned_model_iteration_0.pt", verbose=True
    )
