import torch
import abc
import torch.nn as nn
import lab as B

from . import _dispatch
from .util import batch_size, feature_size

__all__ = ["SetConv", "SetConvPD"]


class AbstractSetConv(nn.Module, metaclass=abc.ABCMeta):
    def __init__(self, num_channels, scale, density):
        if density:
            num_channels += 1
        self.num_channels = num_channels
        self.density = density
        self.log_scales = nn.Parameter(B.log(scale) * B.ones(num_channels))


class SetConv(nn.Module):
    pass


class SetConvPD(nn.Module):
    pass


@_dispatch(SetConv, B.Numeric, B.Numeric, B.Numeric)
def code(layer, xz, z, x, **kw_args):
    pass


@_dispatch(SetConv, type(None), type(None), B.Numeric)
def code(layer, xz, z, x, **kw_args):
    return B.zeros(
        B.device(x),
        B.dtype(x),
        batch_size(x),
        feature_size(x),
        layer.num_channels,
    )
