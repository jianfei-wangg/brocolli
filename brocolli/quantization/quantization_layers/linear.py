import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger

from .base import BaseOperator
from .utils import _quantize_weight, _quantize_bias
from .registry import register_quant_op


@register_quant_op(torch.nn.Linear)
class Linear(nn.Module, BaseOperator):
    def __init__(self, in_features, out_features, bias=True):
        super(Linear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.bias = bias

    def extra_repr(self):
        s = "in_features={in_features}, out_features={out_features}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)

    def _get_name(self):
        return "QuantizedLinear"

    @classmethod
    def from_float(cls, mod):
        assert hasattr(
            mod, "qconfig"
        ), "Conv float module must have qconfig defined."
        weight_observer = mod.qconfig.weight()
        weight_observer(mod.weight)

        qlinear = cls(mod.in_features, mod.out_features, mod.bias)
        qlinear.weight_observer = weight_observer
        qlinear.weight = torch.nn.Parameter(mod.weight, requires_grad=False)
        if mod.bias is not None:
            qlinear.bias = torch.nn.Parameter(mod.bias, requires_grad=False)

        return qlinear

    def forward(self, input):
        out = F.linear(
            input,
            self.weight_observer(self.weight),
            self.bias,
        )

        return out
