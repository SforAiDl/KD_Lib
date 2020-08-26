import torch
from KD_Lib.Quantization.common import Quantizer


class Dynamic_Quantizer(Quantizer):
    """
    Implementation of Dynamic Quantization for PyTorch models.
    
    :param model (torch.nn.Module): Model that needs to be quantized
    """

    def __init__(self, model):
        super(Dynamic_Quantizer, self).__init__(model)

    def quantize(
        self, qconfig_spec=None, dtype=torch.qint8, mapping=None, inplace=False
    ):
        """
		Function used for quantization

		:param qconfig_spec: Either:
            - A dictionary that maps from name or type of submodule to quantization
              configuration, qconfig applies to all submodules of a given
              module unless qconfig for the submodules are specified (when the
              submodule already has qconfig attribute). Entries in the dictionary
              need to be QConfigDynamic instances.
            - A set of types and/or submodule names to apply dynamic quantization to,
              in which case the `dtype` argument is used to specify the bit-width
        :param inplace(bool): carry out model transformations in-place, the original module is mutated
        :param mapping: maps type of a submodule to a type of corresponding dynamically quantized version
            with which the submodule needs to be replaced
        """
        self.quantized_model = torch.quantization.quantize_dynamic(
            self.model,
            qconfig_spec=qconfig_spec,
            dtype=dtype,
            mapping=mapping,
            inplace=inplace,
        )
        return self.quantized_model
