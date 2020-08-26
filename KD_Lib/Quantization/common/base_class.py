import torch
import os
import time


class Quantizer:
    """
    Baisc Implementation of Quantization for PyTorch models.
    :param model (torch.nn.Module): Model that needs to be pruned
    """

    def __init__(self, model):
        self.model = model
        self.quantized_model = model

    def quantize(self):
        """
		Function used for quantization
		"""
        raise NotImplementedError

    def compare_model_sizes(self):
        """
        Function for printing sizes of the original and quantized model
        """
        original_size = self._get_size_of_model(self.model)
        quantized_size = self._get_size_of_model(self.quantized_model)

        print("-" * 80)
        print(f"Size of original model (MB): {original_size}")
        print(f"Size of quantized_model (MB): {quantized_size}")

    def compare_inference_performance(self, data_loader):
        """
        Function used for comparing inference performance of original and quantized models
        Note that performance here referes to the following:
            1. Accuracy achieved on the testset
            2. Time taken for evaluating on the testset
        :param data_loader(torch.utils.data.DataLoader): DataLoader used for evaluation
        """
        acc, elapsed = self._time_model_evaluation(self.model, data_loader)
        print(f"Original Model: Acc: {acc} | Time: {elapsed}s")

        acc, elapsed = self._time_model_evaluation(self.quantized_model, data_loader)
        print(f"Quantized Model: Acc: {acc} | Time: {elapsed}s")

    def _get_size_of_model(self, model):
        torch.save(model.state_dict(), "temp.p")
        model_size = os.path.getsize("temp.p") / 1e6
        os.remove("temp.p")
        return model_size

    def _time_model_evaluation(self, model, data_loader):
        s = time.time()
        acc = self._evaluate_model(model, data_loader)
        elapsed = time.time() - s
        return acc, elapsed

    def _evaluate_model(self, model, data_loader):
        model.eval()
        correct = 0
        len_dataset = len(data_loader.dataset)

        with torch.no_grad():
            for image, target in data_loader:
                output = model(image)

                if isinstance(output, tuple):
                    output = output[0]

                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()

        return correct / len_dataset
