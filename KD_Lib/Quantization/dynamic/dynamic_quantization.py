import torch
from KD_Lib.Quantization.common import Quantizer


class Dynamic_Quantizer(Quantizer):
    """
    Implementation of Dynamic Quantization for PyTorch models.

    :param model (torch.nn.Module): Model that needs to be quantized
    :param qconfig_spec: Qconfig spec
    :param test_loader(torch.utils.data.DataLoader): DataLoader used for testing
    """

    def __init__(
        self,
        model,
        test_loader,
        qconfig_spec=None,
    ):
        super(Dynamic_Quantizer, self).__init__(
            model,
            qconfig_spec,
            test_loader=test_loader,
        )

    def quantize(self, dtype=torch.qint8, mapping=None):
        """
        Function used for quantization

        :param inplace(bool): carry out model transformations in-place, the original module is mutated
        :param mapping: maps type of a submodule to a type of corresponding dynamically quantized version with which the submodule needs to be replaced
        """

        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            qconfig_spec=self.qconfig,
            dtype=dtype,
            mapping=mapping,
            inplace=False,
        )
        return self.quantized_model
