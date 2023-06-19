import torch
import torch.nn as nn
from collections import namedtuple

from .observer import MinMaxObserver, PerChannelMinMaxObserver


class QConfig(namedtuple("QConfig", ["activation", "weight"])):
    def __new__(cls, activation, weight):
        if isinstance(activation, nn.Module) or isinstance(weight, nn.Module):
            raise ValueError(
                "QConfig received observer instance, please pass observer class instead. "
                + "Use MyObserver.with_args(x=1) to override arguments to constructor if needed"
            )
        return super(QConfig, cls).__new__(cls, activation, weight)


def get_qconfig(bit):
    if bit == 8:
        return QConfig(
            activation=MinMaxObserver.with_args(bit=bit),
            weight=PerChannelMinMaxObserver.with_args(bit=bit),
        )
    elif bit == 16:
        return QConfig(
            activation=MinMaxObserver.with_args(bit=bit),
            weight=PerChannelMinMaxObserver.with_args(bit=bit),
        )
    else:
        raise ValueError("Quantization bit {} is not supported".format(bit))
