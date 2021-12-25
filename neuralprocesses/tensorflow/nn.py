from functools import partial
from typing import Optional

import lab.tensorflow as B
import numpy as np
import tensorflow as tf
from plum import convert

from .. import _dispatch

__all__ = ["num_params", "Module"]


@_dispatch
def num_params(x: tf.keras.Model):
    return sum([int(np.prod(p.shape)) for p in x.variables])


class ChannelsToFirst(tf.keras.Model):
    def call(self, x, training=False):
        rank = B.rank(x)
        perm = [0, rank - 1] + list(range(1, rank - 1))
        return B.transpose(x, perm=perm)


class ChannelsToLast(tf.keras.Model):
    def call(self, x, training=False):
        rank = B.rank(x)
        perm = [0] + list(range(2, rank)) + [1]
        return B.transpose(x, perm=perm)


def ConvNd(
    dim: int,
    in_channels: int,
    out_channels: int,
    kernel: int,
    stride: int = 1,
    dilation: int = 1,
    groups: int = 1,
    bias: bool = True,
    transposed: bool = False,
    output_padding: Optional[int] = None,
    dtype=None,
):
    # Only set `output_padding` if it is given.
    additional_args = {}
    if output_padding is not None:
        additional_args["output_padding"] = output_padding

    # Get the right layer kind.
    if transposed:
        suffix = "Transpose"
    else:
        suffix = ""

    # Only set `data_format` on the GPU: there is no CPU support.
    if len(tf.config.list_physical_devices("GPU")) > 0:
        data_format = "channels_first"
    else:
        data_format = "channels_last"

    conv_layer = getattr(tf.keras.layers, f"Conv{dim}D{suffix}")(
        input_shape=(in_channels,) + (None,) * dim,
        filters=out_channels,
        kernel_size=kernel,
        strides=stride,
        padding="same",
        dilation_rate=dilation,
        groups=groups,
        use_bias=bias,
        data_format=data_format,
        dtype=dtype,
        **additional_args,
    )
    if data_format == "channels_first":
        return conv_layer
    else:
        return tf.keras.Sequential(
            [
                ChannelsToLast(dtype=dtype),
                conv_layer,
                ChannelsToFirst(dtype=dtype),
            ]
        )


class Interface:
    ReLU = tf.keras.layers.ReLU

    @staticmethod
    def Sequential(*x):
        return tf.keras.Sequential(x)

    @staticmethod
    def ModuleList(x):
        # TensorFlow tracks regular lists just fine.
        return list(x)

    @staticmethod
    def Linear(dim_in, dim_out, dtype=None):
        return tf.keras.layers.Dense(dim_out, input_shape=(None, dim_in), dtype=dtype)

    Conv1d = partial(ConvNd, dim=1)
    Conv2d = partial(ConvNd, dim=2)
    Conv3d = partial(ConvNd, dim=3)

    ConvTransposed1d = partial(ConvNd, dim=1, transposed=True)
    ConvTransposed2d = partial(ConvNd, dim=2, transposed=True)
    ConvTransposed3d = partial(ConvNd, dim=3, transposed=True)

    @staticmethod
    def Parameter(x, dtype=None):
        dtype = dtype or tf.float32
        dtype = convert(dtype, B.TFDType)
        return tf.Variable(x, dtype=dtype)


interface = Interface()


class Module(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.nn = interface
