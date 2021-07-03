import copy
import numpy as np
import torch
import torch.nn as nn


class BaseIterativePruner:
    """
    Implementation of a Basic Iterative Pruner.
    Any iterative pruning technique can be extended from this by implementing the `prune_model` method.

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
        self.device = device
        self.model = model.to(self.device)
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def train_model(self):
        """
        Function used for training the model for one epoch
        """

        epoch_loss = 0.0
        correct = 0
        length_of_dataset = len(self.train_loader.dataset)

        for (data, label) in self.train_loader:
            data = data.to(self.device)
            label = label.to(self.device)

            out = self.model(data)
            if isinstance(out, tuple):
                out = out[0]

            loss = self.loss_fn(out, label)

            pred = out.argmax(dim=1, keepdim=True)
            correct += pred.eq(label.view_as(pred)).sum().item()

            self.optimizer.zero_grad()
            loss.backward()

            self.zero_pruned_gradients()
            self.optimizer.step()

            epoch_loss += loss.item()

        epoch_loss /= length_of_dataset
        epoch_acc = 100 * (correct / length_of_dataset)
        return epoch_loss, epoch_acc

    def zero_pruned_gradients(self):
        """
        Function used for zeroing gradients of pruned weights
        """

        for name, param in self.model.named_parameters():
            if "weight" in name:
                param_data = param.data.cpu().numpy()
                param_grad = param.grad.data.cpu().numpy()
                param_grad = np.where(param_data == 0.0, 0, param_grad)
                param.grad.data = torch.from_numpy(param_grad).to(self.device)

    def finetune_model(self, epochs, save_model=False, save_model_path="model.pt"):
        """
        Function used for finetuning the model after it is pruned

        :param epochs: Number of training epochs
        :type epochs: int
        :param save_model: True if the model needs to be saved
        :type save_model: bool
        :param save_model_path: Path where the model needs to be saved (only used if save_model = True).
        :type save_model_path: str
        """

        best_acc = 0.0
        best_model_weights = copy.deepcopy(self.model.state_dict())
        loss_arr = []
        accs = []

        for ep in range(epochs):
            epoch_loss, epoch_acc = self.train_model()
            test_loss, test_acc = self.evaluate_model()
            if test_acc > best_acc:
                best_acc = epoch_acc
                best_model_weights = copy.deepcopy(self.model.state_dict())

            loss_arr.append(epoch_loss)
            print(
                f"Epoch: {ep+1}, Training Loss: {epoch_loss}, Training Accuracy: {epoch_acc}"
            )
            print(f"Epoch: {ep+1}, Test Loss: {test_loss}, Test Accuracy: {test_acc}")

        self.model.load_state_dict(best_model_weights)
        if save_model:
            torch.save(self.model.state_dict(), save_model_path)

    def evaluate_model(self, model_path=None):
        """
        Function used for evaluating a model

        :param model_path: Path to a PyTorch model that needs to be evaluated. If None, current final model is used.
        :type model_path: str
        """

        model = copy.deepcopy(self.model)
        if model_path:
            model.load_state_dict(torch.load(model_path))

        model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = model(data)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                test_loss += self.loss_fn(outputs, targets).item()
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(targets.data.view_as(pred)).sum().item()

            test_loss /= len(self.test_loader.dataset)
            test_acc = 100.0 * correct / len(self.test_loader.dataset)

        return test_loss, test_acc

    def prune(
        self, num_iterations=10, train_epochs=10, save_models=True, **prune_params
    ):
        """
        Function used for facilitating the pruning pipeline

        :param num_iterations: Number of iterations for pruning
        :type num_iterations: int
        :param train_epochs: Number of iterations for training per pruning iteration
        :type train_epochs: int
        :param save_models: True if validated models (per pruning iteration) need to be saved
        :type save_models: bool
        :param prune_params: Any additional parameters needed by the "prune_model" method (specific to pruning technique used)
        """

        for it in range(num_iterations):
            print(
                "======== Pruning Iteration {:} / {:} ========".format(
                    it + 1, num_iterations
                )
            )
            if not it == 0:
                self.prune_model(**prune_params)
            self.optimizer = torch.optim.Adam(self.model.parameters())
            self.finetune_model(
                train_epochs, save_models, f"pruned_model_iteration_{it}.pt"
            )

    def prune_model(self, **prune_params):
        """
        Function used to implement the pruning technique.
        Needs to zero parameters of the model that are pruned by the technique.
        """

        raise NotImplementedError

    def get_pruning_statistics(self, model_path=None, verbose=True):
        """
        Function used for priniting layer-wise pruning statistics

        :param model_path: Path of the model whose statistics need to be displayed
                            If None, statistics of the final model is displayed
        :type model_path: str
        :param verbose: If true, the entire statistics is printed
        :type verbose: bool

        :return alive: If model_path is specified, percentage of alive neurons is returned.
                        If model_path is None and saved_models are available, returns a list
                        containing alive neurons percentage for each saved model
                        Else returns -1
        :type alive: int or list
        """

        model = copy.deepcopy(self.model)
        if model_path:
            model.load_state_dict(torch.load(model_path))
        nonzero = total = 0
        for name, p in model.named_parameters():
            tensor = p.data.cpu().numpy()
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            nonzero += nz_count
            total += total_params
            if verbose:
                print(
                    f"{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}"
                )
        if verbose:
            print(
                f"alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)"
            )
        return round((nonzero / total) * 100, 1)
