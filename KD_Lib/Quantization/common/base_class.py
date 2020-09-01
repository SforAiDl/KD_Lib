import torch
import os
import time


class Quantizer:
    """
    Basic Implementation of Quantization for PyTorch models.

    :param model: Model that needs to be pruned
    :type model: torch.nn.Module
    :param qconfig: Configuration used for quantization
    :type qconfig: Qconfig
    :param train_loader: DataLoader used for training
    :type train_loader: torch.utils.data.DataLoader
    :param test_loader: DataLoader used for testing
    :type test_loader: torch.utils.data.DataLoader
    :param optimizer: Optimizer for training
    :type optimizer: torch.optim.*
    :param criterion: Loss function used for calibration
    :type criterion: Loss_fn
    :param device: Device used for training ("cpu" or "cuda")
    :type device: torch.device
    """

    def __init__(
        self,
        model,
        qconfig,
        train_loader=None,
        test_loader=None,
        optimizer=None,
        criterion=None,
        device=torch.device("cpu"),
    ):
        self.model = model
        self.quantized_model = model
        self.qconfig = qconfig
        self.train_loader = train_loader
        self.test_loader = test_loader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

    def quantize(self):
        """
        Function used for quantization
        """

        raise NotImplementedError

    def get_model_sizes(self):
        """
        Function for printing sizes of the original and quantized model
        """

        original_size = self._get_size_of_model(self.model)
        quantized_size = self._get_size_of_model(self.quantized_model)

        print("-" * 80)
        print(f"Size of original model (MB): {original_size}")
        print(f"Size of quantized_model (MB): {quantized_size}")

    def get_performance_statistics(self):
        """
        Function used for reporting inference performance of original and quantized models
        Note that performance here referes to the following:
        1. Accuracy achieved on the testset
        2. Time taken for evaluating on the testset
        """

        acc, elapsed = self._time_model_evaluation(self.model)
        print(f"Original Model: Acc: {acc} | Time: {elapsed}s")

        acc, elapsed = self._time_model_evaluation(self.quantized_model)
        print(f"Quantized Model: Acc: {acc} | Time: {elapsed}s")

    def _get_size_of_model(self, model):
        """
        Function used for fetching size of a model

        :param model: Model
        :type model: torch.nn.Module
        """

        torch.save(model.state_dict(), "temp.p")
        model_size = os.path.getsize("temp.p") / 1e6
        os.remove("temp.p")
        return model_size

    def _time_model_evaluation(self, model):
        """
        Function used for fetching time taken by the model for inference

        :param model: Model
        :type model: torch.nn.Module
        """

        s = time.time()
        acc = self._evaluate_model(model)
        elapsed = time.time() - s
        return acc, elapsed

    def _evaluate_model(self, model):
        """
        Function used for evaluating the model

        :param model: Model
        :type model: torch.nn.Module
        """

        model.eval()
        correct = 0
        len_dataset = len(self.test_loader.dataset)

        with torch.no_grad():
            for image, target in self.test_loader:
                output = model(image)

                if isinstance(output, tuple):
                    output = output[0]

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        return correct / len_dataset
