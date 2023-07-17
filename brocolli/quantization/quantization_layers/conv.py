import torch
import torch.nn as nn
import torch.nn.functional as F

from loguru import logger

from .base import BaseOperator
from .utils import _pair, _quantize_weight, _quantize_bias
from .registry import register_quant_op

_SUPPORTED_PADDING = {"zeros", "reflect"}


class _ConvNd(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        dtype=None,
    ):
        raise NotImplementedError

    def _init(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride,
        padding,
        dilation,
        transposed,
        output_padding,
        groups,
        bias,
        padding_mode="zeros",
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype}
        super(_ConvNd, self).__init__()

        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = transposed
        self.output_padding = output_padding
        self.groups = groups
        if padding_mode not in _SUPPORTED_PADDING:
            raise ValueError(
                "'padding_mode' {} is not supported by quantized convolution".format(
                    padding_mode
                )
            )
        self.padding_mode = padding_mode
        self.scale = 1.0

    def extra_repr(self):
        s = (
            "{in_channels}, {out_channels}, kernel_size={kernel_size}"
            ", stride={stride}"
        )
        if self.padding != (0,) * len(self.padding):
            s += ", padding={padding}"
        if self.dilation != (1,) * len(self.dilation):
            s += ", dilation={dilation}"
        if self.output_padding != (0,) * len(self.output_padding):
            s += ", output_padding={output_padding}"
        if self.groups != 1:
            s += ", groups={groups}"
        if self.bias is None:
            s += ", bias=False"
        return s.format(**self.__dict__)

    def __copy__(self):
        return self.__deepcopy__({})

    @classmethod
    def get_qconv(
        cls,
        mod,
    ):
        weight_observer = mod.qconfig.weight()
        weight_observer(mod.weight)

        qconv = cls(
            mod.in_channels,
            mod.out_channels,
            mod.kernel_size,
            mod.stride,
            mod.padding,
            mod.dilation,
            mod.groups,
            None,
            mod.padding_mode,
        )
        qconv.name = mod.name
        qconv.weight_observer = weight_observer
        qconv.weight = torch.nn.Parameter(mod.weight, requires_grad=False)
        if mod.bias is not None:
            qconv.bias = torch.nn.Parameter(mod.bias, requires_grad=False)

        return qconv

    @staticmethod
    def from_float(cls, mod):
        assert hasattr(
            mod, "qconfig"
        ), "Conv float module must have qconfig defined."

        return cls.get_qconv(mod)


@register_quant_op(torch.nn.Conv2d)
class Conv2d(_ConvNd, BaseOperator):
    def __init__(
        self,
        in_channels,
        out_channels,
        kernel_size,
        stride=1,
        padding=0,
        dilation=1,
        groups=1,
        bias=True,
        padding_mode="zeros",
        dtype=None,
    ):
        factory_kwargs = {"dtype": dtype}
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)

        super(Conv2d, self)._init(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            False,
            _pair(0),
            groups,
            bias,
            padding_mode,
            **factory_kwargs,
        )

    def _get_name(self):
        return "TransformedConv2d"

    def forward(self, input):
        out = F.conv2d(
            input,  # double may overflow for large input
            self.weight_observer(self.weight),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )        

        return out


    @classmethod
    def from_float(cls, mod):
        r"""Creates a quantized module from a float module or qparams_dict.

        Args:
            mod (Module): a float module, either produced by torch.quantization
              utilities or provided by the user
        """
        return _ConvNd.from_float(cls, mod)
