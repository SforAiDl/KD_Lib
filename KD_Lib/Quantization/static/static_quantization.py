import torch
from KD_Lib.Quantization.common import Quantizer
from copy import deepcopy


class Static_Quantizer(Quantizer):
    """
    Implementation of Static Quantization for PyTorch models.

    :param model (torch.nn.Module): (Quantizable) Model that needs to be quantized
    """

    def __init__(self, model):
        super(Static_Quantizer, self).__init__(model)

    def quantize(
        self,
        data_loader,
        criterion,
        num_calibration_batches=10,
        qconfig=torch.quantization.default_qconfig,
    ):
        """
                Function used for quantization

        :param data_loader(torch.utils.data.DataLoader): DataLoader used for calibration
        :param criterion(torch Loss_fn): Loss function used for calibration
        :param num_calibration_batches(int): Number of batches used for calibration
        :param qconfig: Configuration used for quantization
        """
        self.quantized_model = deepcopy(self.model)
        self.quantized_model.eval()
        self.quantized_model.fuse_model()
        self.quantized_model.qconfig = qconfig

        torch.quantization.prepare(self.quantized_model, inplace=True)

        print("Calibrating model...")
        self._calibrate_model(data_loader, num_calibration_batches, criterion)

        print("Converting to quantized model...")
        torch.quantization.convert(self.quantized_model, inplace=True)

        return self.quantized_model

    def _calibrate_model(self, data_loader, num_batches, criterion):
        """
        Function used for calibrating the model for quantization

        :param data_loader(torch.utils.data.DataLoader): DataLoader used for calibration
        :param num_batches(int): Number of batches used for calibration
        :param criterion(torch Loss_fn): Loss function used for calibration
        """

        self.quantized_model.eval()
        correct = 0
        cnt = 0
        len_dataset = min(
            num_batches * data_loader.batch_size, len(data_loader.dataset)
        )

        with torch.no_grad():
            for image, target in data_loader:
                output = self.quantized_model(image)

                if isinstance(output, tuple):
                    output = output[0]

                loss = criterion(output, target)
                cnt += 1
                pred = output.argmax(dim=1, keepdim=True)
                correct += pred.eq(target.view_as(pred)).sum().item()
                if cnt >= num_batches:
                    return correct / len_dataset

        return correct / len_dataset
