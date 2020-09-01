import copy
import numpy as np
import torch
import torch.nn as nn
import os


class Lottery_Tickets_Pruner:
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

        self.device = device
        self.model = model.to(self.device)
        self.initial_state_dict = copy.deepcopy(self.model.state_dict())
        self.mask = self._initialize_mask()
        self.loss_fn = loss_fn
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = torch.optim.Adam(self.model.parameters())

    def prune(
        self,
        prune_percent=10,
        num_iterations=10,
        train_iterations=10,
        valid_freq=10,
        print_freq=10,
        save_models=False,
    ):
        """
        Function used for pruning

        :param prune_percent: Pruning percent
        :type prune_percent: int
        :param num_iterations: Number of iterations for pruning
        :type num_iterations: int
        :param train_iterations: Number of iterations for training per pruning iteration
        :type train_iterations: int
        :param valid_freq: Frequency of testing (only these models can be stored using save_models)
        :type valid_freq: int
        :param print_freq: Frequency of printing training results
        :type print_freq: int
        :param save_models: True if validated models need to be saved
        :type save_models: bool
        """

        self.num_iterations = num_iterations
        self.train_iterations = train_iterations
        self.percent = prune_percent
        self.valid_freq = valid_freq
        self.print_freq = print_freq
        self.save_models = save_models
        self.saved_models = []

        for it in range(self.num_iterations):
            print(f"Iteration {it}...")
            if not it == 0:
                self._prune_by_percentile()
                self._original_initialization()
            self.optimizer = torch.optim.Adam(self.model.parameters())

            self._train_after_pruning(it)

    def _initialize_mask(self):
        step = 0
        # print("Initializing mask...")
        for name, param in self.model.named_parameters():
            if "weight" in name:
                step += 1

        mask = [None] * step
        step = 0

        for name, param in self.model.named_parameters():
            if "weight" in name:
                param_data = param.data.cpu().numpy()
                mask[step] = np.ones_like(param_data)
                step += 1

        return mask

    def _prune_by_percentile(self):
        step = 0

        for name, param in self.model.named_parameters():
            if "weight" in name:
                param_data = param.data.cpu().numpy()
                alive = param_data[np.nonzero(param_data)]
                percentile = np.percentile(abs(alive), self.percent)
                new_mask = np.where(abs(param_data) < percentile, 0, self.mask[step])
                param.data = torch.from_numpy(param_data * new_mask).to(param.device)
                self.mask[step] = new_mask
                step += 1

    def _original_initialization(self):
        step = 0

        for name, param in self.model.named_parameters():
            if "weight" in name:
                param_data = (
                    self.mask[step] * self.initial_state_dict[name].cpu().numpy()
                )
                param.data = torch.from_numpy(param_data).to(param.device)
                step += 1
            if "bias" in name:
                param.data = self.initial_state_dict[name]

    def _train_after_pruning(self, prune_it):
        best_acc = 0.0
        best_weights = copy.deepcopy(self.model.state_dict())
        losses = []
        accs = []

        for it in range(self.train_iterations):
            # print("Training model...")
            loss, acc = self._train_pruned_model()
            losses.append(loss)
            accs.append(acc)

            if (it % self.print_freq) == 0:
                print(
                    f"Train Epoch: {it}/{self.train_iterations} Loss: {loss:.6f} Training Accuracy: {acc:.2f}% Best Validation Accuracy: {best_acc:.2f}%"
                )

            if (it % self.valid_freq) == 0:
                # print(f'Iteration {it} testing...')
                _, acc = self._test_pruned_model()
                # print(f"Accuracy: {acc}")
                if acc > best_acc:
                    # print("Accuracy better than best!")
                    best_acc = acc
                    if self.save_models:
                        best_weights = copy.deepcopy(self.model.state_dict())

        self._save_model(prune_it, best_weights)

    def _test_pruned_model(self):
        self.model.eval()
        test_loss = 0
        correct = 0

        with torch.no_grad():
            for data, targets in self.test_loader:
                data, targets = data.to(self.device), targets.to(self.device)
                outputs = self.model(data)

                if isinstance(outputs, tuple):
                    outputs = outputs[0]

                test_loss += self.loss_fn(outputs, targets).item()
                pred = outputs.argmax(dim=1, keepdim=True)
                correct += pred.eq(targets.data.view_as(pred)).sum().item()

            test_loss /= len(self.test_loader.dataset)
            test_acc = 100.0 * correct / len(self.test_loader.dataset)

        return test_loss, test_acc

    def _train_pruned_model(self):
        eps = 1e-6
        self.model.train()
        correct = 0

        step = 0
        for data, targets in self.train_loader:
            self.optimizer.zero_grad()
            data, targets = data.to(self.device), targets.to(self.device)
            outputs = self.model(data)

            if isinstance(outputs, tuple):
                outputs = outputs[0]

            train_loss = self.loss_fn(outputs, targets)
            train_loss.backward()

            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.data.view_as(pred)).sum().item()

            train_acc = 100.0 * correct / len(self.train_loader.dataset)

            # print(f"Training Step: {step} | Loss: {train_loss.item()} | Accuracy: {train_acc}")

            for name, param in self.model.named_parameters():
                if "weight" in name:
                    param_data = param.data.cpu().numpy()
                    param_grad = param.grad.data.cpu().numpy()
                    param_grad = np.where(param_data < eps, 0, param_grad)
                    param.grad.data = torch.from_numpy(param_grad).to(self.device)

            self.optimizer.step()
            step += 1

        train_acc = 100.0 * correct / len(self.train_loader.dataset)
        return train_loss.item(), train_acc

    def _save_model(self, prune_it, best_weights):
        file_name = f"{os.getcwd()}/pruned_model_{prune_it}.pth.tar"
        self.saved_models.append(file_name)
        self.model.load_state_dict(best_weights)
        torch.save(self.model, file_name)

    def get_pruning_statistics(self, model_path=None):
        """
        Function used for priniting layer-wise pruning statistics

        :param model_path: Path of the model whose statistics need to be displayed
                            If None, statistics of all the saved models is displayed
        :type model_path: str

        :return alive: If model_path is specified, percentage of alive neurons is returned.
                        If model_path is None and saved_models are available, returns a list
                        containing alive neurons percentage for each saved model
                        Else returns -1
        :type alive: int or list
        """

        if model_path is not None:
            alive = _get_pruning_statistics(model_path)
        else:
            if len(self.saved_models) == 0:
                print("No saved models found.")
                alive = -1
            else:
                alive = []
                for path in self.save_models:
                    print(f"Model: {path}")
                    alive.append(_get_pruning_statistics(path))
        return alive

    def _get_pruning_statistics(self, model_path=None):
        nonzero = total = 0
        for name, p in model.named_parameters():
            tensor = p.data.cpu().numpy()
            nz_count = np.count_nonzero(tensor)
            total_params = np.prod(tensor.shape)
            nonzero += nz_count
            total += total_params
            print(
                f"{name:20} | nonzeros = {nz_count:7} / {total_params:7} ({100 * nz_count / total_params:6.2f}%) | total_pruned = {total_params - nz_count :7} | shape = {tensor.shape}"
            )
        print(
            f"alive: {nonzero}, pruned : {total - nonzero}, total: {total}, Compression rate : {total/nonzero:10.2f}x  ({100 * (total-nonzero) / total:6.2f}% pruned)"
        )
        return round((nonzero / total) * 100, 1)
