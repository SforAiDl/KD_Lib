import torch
import os


class Dynamic_Quantizer:
    """
    Implementation of Dynamic Quantization for PyTorch models.
    :param model (torch.nn.Module): Model that needs to be pruned
    """

    def __init__(self, model):
        self.model = model
        self.quantized_model = model

    def quantize(self, layers, dtype=torch.qint8):
        """
		Function used for quantization
		:param layers (list or tuple): Layer types that need to be quantized (for example, nn.Linear)
		:param dtype (torch.dtype): Dtype of the layers are quantization (torch.qint8 by default)
		"""
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model, layers, dtype=dtype
        )
        return self.quantized_model

    def compare_model_sizes(self):
        """
        Function for printing sizes of the original and quantized model
        """
        original_size = self._get_size_of_model(self.model)
        quantized_size = self._get_size_of_model(self.quantized_model)

        print("-" * 80)
        print(f"Size of original model (MB): {original_size}")
        print(f"Size of quantized_model (MB): {quantized_size}")

    def _get_size_of_model(self, model):
        torch.save(model.state_dict(), "temp.p")
        model_size = os.path.getsize("temp.p") / 1e6
        os.remove("temp.p")
        return model_size
