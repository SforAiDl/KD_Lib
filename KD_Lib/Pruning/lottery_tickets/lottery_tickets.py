import copy
import numpy as np
import torch
import torch.nn as nn

from ..common import BaseIterativePruner


class LotteryTicketsPruner(BaseIterativePruner):
    """
    Implementation of Lottery Tickets Pruning for PyTorch models.

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

        self.initial_state_dict = copy.deepcopy(self.model.state_dict())

    def prune_model(self, prune_percent=10):
        """
        Function used for pruning

        :param prune_percent: Pruning percent per iteration (percentage of alive weights to zero per pruning iteration)
        :type prune_percent: int
        """

        for name, param in self.model.named_parameters():
            if "weight" in name:
                param_data = param.data.cpu().numpy()
                alive = param_data[np.nonzero(param_data)]
                percentile = np.percentile(abs(alive), prune_percent)
                new_param_data = np.where(
                    abs(param_data) < percentile, 0, self.initial_state_dict[name]
                )
                param.data = torch.from_numpy(new_param_data).to(param.device)
            if "bias" in name:
                param.data = self.initial_state_dict[name]
