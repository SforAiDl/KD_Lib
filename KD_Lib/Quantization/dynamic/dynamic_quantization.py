import torch
from KD_Lib.Quantization.common import Quantizer


class Dynamic_Quantizer(Quantizer):
    """
    Implementation of Dynamic Quantization for PyTorch models.

    :param model: Model that needs to be quantized
    :type model: torch.nn.Module
    :param qconfig_spec: Qconfig spec
    :type qconfig_spec: Qconfig_spec
    :param test_loader: DataLoader used for testing
    :type test_loader: torch.utils.data.DataLoader
    """

    def __init__(self, model, test_loader, qconfig_spec=None):
        super(Dynamic_Quantizer, self).__init__(
            model, qconfig_spec, test_loader=test_loader
        )

    def quantize(self, dtype=torch.qint8, mapping=None):
        """
        Function used for quantization

        :param dtype: dtype for quantized modules
        :type dtype: torch.dtype
        :param mapping: maps type of a submodule to a type of corresponding dynamically quantized version with which the submodule needs to be replaced
        :type mapping: mapping
        """

        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            qconfig_spec=self.qconfig,
            dtype=dtype,
            mapping=mapping,
            inplace=False,
        )
        return self.quantized_model
