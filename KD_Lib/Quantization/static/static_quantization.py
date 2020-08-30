import torch
from KD_Lib.Quantization.common import Quantizer
from copy import deepcopy


class Static_Quantizer(Quantizer):
    """
    Implementation of Static Quantization for PyTorch models.

    :param model (torch.nn.Module): (Quantizable) Model that needs to be quantized
    :param qconfig: Configuration used for quantization
    :param train_loader(torch.utils.data.DataLoader): DataLoader used for calibration
    :param test_loader(torch.utils.data.DataLoader): DataLoader used for testing
    :param criterion(torch Loss_fn): Loss function used for calibration
    :param device(torch.device): Device used for training ("cpu" or "cuda")
    """

    def __init__(
        self,
        model,
        train_loader,
        test_loader,
        qconfig=torch.quantization.default_qconfig,
        criterion=torch.nn.CrossEntropyLoss(),
        device=torch.device("cpu"),
    ):
        super(Static_Quantizer, self).__init__(
            model, qconfig, train_loader, test_loader, None, criterion, device
        )

    def quantize(self, num_calibration_batches=10):
        """
        Function used for quantization

        :param num_calibration_batches(int): Number of batches used for calibration
        """

        self.quantized_model = deepcopy(self.model)
        self.quantized_model.eval()
        self.quantized_model.fuse_model()
        self.quantized_model.qconfig = self.qconfig

        torch.quantization.prepare(self.quantized_model, inplace=True)

        print("Calibrating model...")
        self._calibrate_model(num_calibration_batches)

        print("Converting to quantized model...")
        torch.quantization.convert(self.quantized_model, inplace=True)

        return self.quantized_model

    def _calibrate_model(self, num_batches):
        """
        Function used for calibrating the model for quantization

        :param num_batches(int): Number of batches used for calibration
        """

        self.quantized_model.eval()
        correct = 0
        cnt = 0
        len_dataset = min(
            num_batches * self.train_loader.batch_size, len(self.train_loader.dataset)
        )

        with torch.no_grad():
            for image, target in self.train_loader:
                output = self.quantized_model(image)

                if isinstance(output, tuple):
                    output = output[0]

                loss = self.criterion(output, target)
                cnt += 1
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                if cnt >= num_batches:
                    return correct / len_dataset

        return correct / len_dataset
