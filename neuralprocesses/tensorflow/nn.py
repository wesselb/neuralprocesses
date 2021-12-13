from functools import partial
from typing import Optional

import lab.tensorflow as B
import tensorflow as tf

__all__ = ["Module"]


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
        kernel_size=kernel_size,
        strides=stride,
        padding="same",
        dilation_rate=dilation,
        groups=groups,
        use_bias=bias,
        data_format=data_format,
        **additional_args,
    )
    if data_format == "channels_first":
        return conv_layer
    else:
        return tf.keras.Sequential([ChannelsToLast(), conv_layer, ChannelsToFirst()])


class Interface:
    ReLU = tf.keras.layers.ReLU

    @staticmethod
    def Sequential(*x):
        return tf.keras.Sequential(x)

    @staticmethod
    def Linear(dim_in, dim_out):
        return tf.keras.layers.Dense(dim_out, input_shape=(None, dim_in))

    Conv1d = partial(ConvNd, dim=1)
    Conv2d = partial(ConvNd, dim=2)
    Conv3d = partial(ConvNd, dim=3)

    ConvTransposed1d = partial(ConvNd, dim=1, transposed=True)
    ConvTransposed2d = partial(ConvNd, dim=2, transposed=True)
    ConvTransposed3d = partial(ConvNd, dim=3, transposed=True)


interface = Interface()


class Module(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.nn = interface