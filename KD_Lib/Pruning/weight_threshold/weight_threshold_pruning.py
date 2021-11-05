import torch
import torch.nn as nn

from ..common import BaseIterativePruner


class WeightThresholdPruner(BaseIterativePruner):
    """
    Implementation of Weight Threshold Pruning for PyTorch models.
        Prunes weights with magnitudes lesser than the specified threshold.

    :param model: Model that needs to be pruned
    :type model: torch.nn.Module
    :param train_loader: Dataloader for training
    :type train_loader: torch.utils.data.DataLoader
    :param test_loader: Dataloader for validation/testing
    :type test_loader: torch.utils.data.DataLoader
    :param loss_fn: Loss function to be used for training
    :type loss_fn: torch.nn.Module
    :param device: Device used for implementation ("cpu" by default)
    :type device: torch.device
    """

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        loss_fn=nn.CrossEntropyLoss(),
        device="cpu",
    ):
        super().__init__(model, train_loader, test_loader, loss_fn, device)

    def prune_model(self, threshold):
        """
        Function used for pruning

        :param threshold: Weight threshold. Weights with magnitudes lesser than the threshold are pruned.
            :type threshold: float
        """

        for name, param in self.model.named_parameters():
            if "weight" in name:
                param_mask = torch.abs(param) < threshold
                param.data[param_mask] = 0.0
