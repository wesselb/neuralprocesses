from functools import partial
from typing import Optional

import torch

__all__ = ["Module"]


def ConvNd(
    dim: int,
    in_channels: int,
    out_channels: int,
    kernel_size: int,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    transposed: bool = False,
    output_padding: Optional[int] = None,
):
    # Only set `output_padding` if it is given.
    additional_args = {}
    if output_padding is not None:
        additional_args["output_padding"] = output_padding

    # Use same-padding.
    if kernel_size % 2 != 1:
        raise ValueError("Kernel size must be odd to achieve same-padding.")

    # Get the right layer kind.
    if transposed:
        suffix = "Transpose"
    else:
        suffix = ""

    return getattr(torch.nn, f"Conv{suffix}{dim}d")(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=kernel_size // 2,
        dilation=dilation,
        groups=groups,
        bias=bias,
        **additional_args,
    )


class Interface:
    ReLU = torch.nn.ReLU

    Sequential = torch.nn.Sequential

    Linear = torch.nn.Linear

    Conv1d = partial(ConvNd, dim=1)
    Conv2d = partial(ConvNd, dim=2)
    Conv3d = partial(ConvNd, dim=3)

    ConvTransposed1d = partial(ConvNd, dim=1, transposed=True)
    ConvTransposed2d = partial(ConvNd, dim=2, transposed=True)
    ConvTransposed3d = partial(ConvNd, dim=3, transposed=True)


interface = Interface()


class Module(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.nn = interface
