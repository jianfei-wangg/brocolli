from abc import ABCMeta, abstractmethod
from typing import Any
from functools import partial
import torch
import torch.nn as nn

ABC: Any = ABCMeta(str("ABC"), (object,), {})

AVIAIABLE_OBSERVERS = []


def register_observer(cls):
    AVIAIABLE_OBSERVERS.append(cls)

    return cls


def get_available_observers():
    return AVIAIABLE_OBSERVERS


def _with_args(cls_or_self, **kwargs):
    class _PartialWrapper(object):
        def __init__(self, p):
            self.p = p

        def __call__(self, *args, **keywords):
            return self.p(*args, **keywords)

        def __repr__(self):
            return self.p.__repr__()

        with_args = _with_args

    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    return r


class ObserverBase(ABC, nn.Module):
    def __init__(self, bit):
        super(ObserverBase, self).__init__()
        self.bit = bit

    @abstractmethod
    def forward(self, x):
        pass

    @abstractmethod
    def calculate_qparams(self, **kwargs):
        pass

    with_args = classmethod(_with_args)


class _ObserverBase(ObserverBase):
    _version = 2

    eps: torch.Tensor

    def __init__(
        self,
        bit=8,
    ):
        super(_ObserverBase, self).__init__(bit=bit)
        self.register_buffer(
            "eps", torch.tensor([torch.finfo(torch.float32).eps])
        )
        self.input_shape = None
        self.lsq_enabled = False

    def grad_scale(self, x, scale):
        y = x
        y_grad = x * scale
        return (y - y_grad).detach() + y_grad

    def round_pass(self, x):
        y = x.round()
        y_grad = x
        return (y - y_grad).detach() + y_grad

    def _lsq_forward(self, x):
        s_grad_scale = 1.0 / ((self.quant_max * x.numel()) ** 0.5)
        s_scale = self.grad_scale(self.lsq_scale, s_grad_scale)

        x = x / s_scale
        x = torch.clamp(x, self.quant_min, self.quant_max)
        x = self.round_pass(x)
        x = x * s_scale
        return x

    @torch.jit.export
    def _calculate_qmin_qmax(self):
        quant_min, quant_max = -127, 127

        return quant_min, quant_max

    def enable_lsq(self):
        self.lsq_scale = nn.Parameter(self.scale)
        self.lsq_enabled = True

    def disable_lsq(self):
        self.lsq_scale = self.scale.detach().data
        self.lsq_enabled = False


@register_observer
class MinMaxObserver(_ObserverBase):
    min_val: torch.Tensor
    max_val: torch.Tensor

    def __init__(
        self, bit=8,
    ):
        super(MinMaxObserver, self).__init__(bit=bit,
        )
        self.register_buffer("min_val", torch.tensor(float("inf")))
        self.register_buffer("max_val", torch.tensor(float("-inf")))

    def forward(self, x_orig):
        if self.lsq_enabled:
            return self._lsq_forward(x_orig)

        self.input_shape = x_orig.shape
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        x = x.to(self.min_val.dtype)
        min_val_cur, max_val_cur = torch.aminmax(x)
        min_val = torch.min(min_val_cur, self.min_val)
        max_val = torch.max(max_val_cur, self.max_val)
        self.min_val.copy_(min_val)
        self.max_val.copy_(max_val)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        if self.lsq_enabled:
            return self.lsq_scale

        self.quant_min, self.quant_max = self._calculate_qmin_qmax()
        if self.min_val >= 0:
            self.quant_min, self.quant_max = 0, 255

        max_val_pos = torch.max(torch.abs(self.min_val), self.max_val)
        self.scale = max_val_pos / self.quant_max
        return self.scale

    @torch.jit.export
    def extra_repr(self):
        return "min_val={}, max_val={}".format(self.min_val, self.max_val)


@register_observer
class PerChannelMinMaxObserver(_ObserverBase):
    min_vals: torch.Tensor
    max_vals: torch.Tensor

    def __init__(self, ch_axis=0, bit=8):
        super(PerChannelMinMaxObserver, self).__init__(bit=bit)
        self.ch_axis = ch_axis
        self.register_buffer("min_vals", torch.tensor([]))
        self.register_buffer("max_vals", torch.tensor([]))

    def forward(self, x_orig):
        self.input_shape = x_orig.shape
        return self._forward(x_orig)

    def _forward(self, x_orig):
        if x_orig.numel() == 0:
            return x_orig
        x = x_orig.detach()  # avoid keeping autograd tape
        min_vals = self.min_vals
        max_vals = self.max_vals
        x_dim = x.size()

        new_axis_list = [i for i in range(len(x_dim))]  # noqa: C416
        new_axis_list[self.ch_axis] = 0
        new_axis_list[0] = self.ch_axis
        y = x.permute(new_axis_list)
        y = y.to(self.min_vals.dtype)
        y = torch.flatten(y, start_dim=1)
        if min_vals.numel() == 0 or max_vals.numel() == 0:
            min_vals, max_vals = torch._aminmax(y, 1)
        else:
            min_vals_cur, max_vals_cur = torch._aminmax(y, 1)
            min_vals = torch.min(min_vals_cur, min_vals)
            max_vals = torch.max(max_vals_cur, max_vals)
        self.min_vals.resize_(min_vals.shape)
        self.max_vals.resize_(max_vals.shape)
        self.min_vals.copy_(min_vals)
        self.max_vals.copy_(max_vals)
        return x_orig

    @torch.jit.export
    def calculate_qparams(self):
        quant_min, quant_max = self._calculate_qmin_qmax()
        min_val_neg = torch.min(self.min_vals, torch.zeros_like(self.min_vals))
        max_val_pos = torch.max(self.max_vals, torch.zeros_like(self.max_vals))

        scale = torch.ones(min_val_neg.size(), dtype=torch.float32)

        max_val_pos = torch.max(-min_val_neg, max_val_pos)
        scale = max_val_pos / (float(quant_max - quant_min) / 2)
        scale = torch.max(scale, self.eps)

        if len(scale.shape) == 0:
            scale = torch.tensor([float(scale)], dtype=scale.dtype)

        return scale

    def extra_repr(self):
        return "min_val={}, max_val={}".format(self.min_vals, self.max_vals)


default_observer = MinMaxObserver.with_args()
default_weight_observer = PerChannelMinMaxObserver.with_args(bit=8)
