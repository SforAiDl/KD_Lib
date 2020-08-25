import torch


class Dynamic_Quantizer:
    """
    Implementation of Dynamic Quantization for PyTorch models.
    :param model (torch.nn.Module): Model that needs to be pruned
    """

    def __init__(self, model):
        self.model = model

    def quantize(self, layers, dtype=torch.qint8):
        """
		Function used for quantization
		:param layers (list or tuple): Layer types that need to be quantized (for example, nn.Linear)
		:param dtype (torch.dtype): Dtype of the layers are quantization (torch.qint8 by default)
		"""
        quantized_model = torch.quantization.quantize_dynamic(
            self.model, layers, dtype=dtype
        )
        return quantized_model
