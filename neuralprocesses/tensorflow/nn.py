from functools import partial
from typing import Optional, Union

import lab.tensorflow as B
import numpy as np
import tensorflow as tf
from plum import convert

import neuralprocesses as nps
from .. import _dispatch
from ..util import compress_batch_dimensions, is_framework_module

__all__ = ["num_params", "Module"]


@is_framework_module.dispatch
def is_framework_module(x: Union[tf.keras.Model, tf.keras.layers.Layer]):
    # Register TF framework types.
    return True


@_dispatch
def num_params(model: tf.keras.Model):
    """Get the number of parameters.

    Args:
        model (:class:`tf.keras.Model`): Keras model.

    Returns:
        int: Number of parameters.
    """
    return sum([int(np.prod(p.shape)) for p in model.variables])


class ChannelsToFirst(tf.keras.Model):
    """Convert from channels last format to channels first format."""

    def call(self, x, training=False):
        rank = B.rank(x)
        perm = [0, rank - 1] + list(range(1, rank - 1))
        return B.transpose(x, perm=perm)


class ChannelsToLast(tf.keras.Model):
    """Convert from channels first format to channels last format."""

    def call(self, x, training=False):
        rank = B.rank(x)
        perm = [0] + list(range(2, rank)) + [1]
        return B.transpose(x, perm=perm)


class CompressBatchDimensions(tf.keras.Model):
    """Compress batch dimensions.

    Args:
        module (module): Module to wrap.
        other_dims (int): Number of other dimensions.
        dtype (dtype, optional): Data type.
    """

    def __init__(self, module, other_dims, dtype=None):
        super().__init__(dtype=dtype)
        self.module = module
        self.other_dims = other_dims

    def call(self, x, training=False):
        x, uncompress = compress_batch_dimensions(x, self.other_dims)
        x = self.module(x)
        return uncompress(x)


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
    """Convolutional layer.

    Args:
        dim (int): Dimensionality.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        kernel (int): Kernel size.
        stride (int, optional): Stride.
        dilation (int, optional): Dilation.
        groups (int, optional): Number of groups.
        bias (bool, optional): Use a bias. Defaults to `True`.
        transposed (bool, optional): Transposed convolution. Defaults to `False`.
        output_padding (int, optional): Output padding.
        dtype (dtype, optional): Data type.

    Returns:
        object: Convolutional layer.
    """
    # Only set `data_format` on the GPU: there is no CPU support.
    if len(tf.config.list_physical_devices("GPU")) > 0:
        data_format = "channels_first"
    else:
        data_format = "channels_last"

    # Only set `output_padding` if it is given.
    additional_args = {}
    if output_padding is not None:
        additional_args["output_padding"] = output_padding

    # Get the right layer kind.
    if transposed:
        suffix = "Transpose"
    else:
        suffix = ""

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


def UpSamplingNd(
    dim: int,
    size: int = 2,
    interp_method: str = "nearest",
    dtype=None,
):
    """Up-sampling layer.

    Args:
        dim (int): Dimensionality.
        size (int, optional): Up-sampling factor. Defaults to `2`.
        interp_method (str, optional): Interpolation method. Can be set to "bilinear".
            Defaults to "nearest'.
        dtype (dtype): Data type.

    Returns:
        object: Up-sampling layer.
    """
    # Only set `data_format` on the GPU: there is no CPU support. Moreover,
    # `UpSampling1D` does not accept the keyword argument `data_format`.
    if len(tf.config.list_physical_devices("GPU")) > 0 and dim > 1:
        data_format = "channels_first"
    else:
        data_format = "channels_last"

    # Due to inconsistent input signatures between TensorFlow's `UpSamplingND`
    # classes, we must construct keyword arguments dependent on the input dimension.
    additional_args = {}
    if dim > 1:
        # `UpSampling1D` does not accept the keyword `data_format` and requires data as
        # channels last.
        additional_args["data_format"] = data_format
    if dim == 2:
        # Only `UpSampling2D` accepts the `interpolation` keyword argument; the 1D and
        # 3D classes only supports nearest-neighbour interpolation.
        additional_args["interpolation"] = interp_method
    elif interp_method != "nearest":
        raise ValueError(
            f'With {dim}D inputs, only "nearest" is supported for `interp_method`.'
        )

    upsample_layer = getattr(tf.keras.layers, f"UpSampling{dim}D")(
        size=size if dim == 1 else (size,) * dim,
        dtype=dtype,
        **additional_args,
    )
    if data_format == "channels_first":
        return upsample_layer
    else:
        return tf.keras.Sequential(
            [
                ChannelsToLast(dtype=dtype),
                upsample_layer,
                ChannelsToFirst(dtype=dtype),
            ]
        )


def AvgPoolNd(
    dim: int,
    kernel: int,
    stride: Union[None, int] = None,
    dtype=None,
):
    """Average pooling layer.

    Args:
        dim (int): Dimensionality.
        kernel (int): Kernel size.
        stride (int, optional): Stride.
        dtype (dtype): Data type.

    Returns:
        object: Average pooling layer.
    """
    # Only set `data_format` on the GPU: there is no CPU support.
    if len(tf.config.list_physical_devices("GPU")) > 0:
        data_format = "channels_first"
    else:
        data_format = "channels_last"

    pool_layer = getattr(tf.keras.layers, f"AveragePooling{dim}D")(
        pool_size=kernel,
        strides=stride,
        padding="valid",
        data_format=data_format,
        dtype=dtype,
    )

    if data_format == "channels_first":
        return pool_layer
    else:
        return tf.keras.Sequential(
            [
                ChannelsToLast(dtype=dtype),
                pool_layer,
                ChannelsToFirst(dtype=dtype),
            ]
        )


def LayerNorm(*sizes: Union[int, None], dtype=None):
    """Layer normalisation.

    Args:
        *sizes (int or None): Sizes of the final dimensions to normalise. Set a size
            to `None` if it need not be normalised.
        dtype (dtype): Data type.

    Returns:
        object: Layer normalisation.
    """
    return CompressBatchDimensions(
        tf.keras.layers.LayerNormalization(
            [-len(sizes) + i for i, s in enumerate(sizes) if s is not None],
            dtype=dtype,
        ),
        len(sizes),
        dtype=dtype,
    )


class Interface:
    """TensorFlow interface."""

    ReLU = tf.keras.layers.ReLU

    @staticmethod
    def Sequential(*modules):
        """Put modules in a sequence.

        Args:
            *modules (object): Modules.

        Returns:
            :class:`tf.keras.Sequential`: `modules` in sequence.
        """
        return tf.keras.Sequential(modules)

    @staticmethod
    def ModuleList(modules):
        """Make a list of modules whose parameters are tracked.

        Args:
            modules (list): List of modules.

        Returns:
            list: List of `modules` whose parameters are tracked.
        """
        # TensorFlow tracks regular lists just fine.
        return list(modules)

    @staticmethod
    def Linear(dim_in: int, dim_out: int, dtype=None):
        """A linear layer.

        Args:
            dim_in (int): Input dimensionality.
            dim_out (int): Output dimensionality.
            dtype (dtype, optional): Data type.

        Returns:
            :class:`tf.keras.Dense`: Linear layer.
        """
        return tf.keras.layers.Dense(dim_out, input_shape=(None, dim_in), dtype=dtype)

    Conv = staticmethod(ConvNd)
    Conv1d = partial(ConvNd, dim=1)
    Conv2d = partial(ConvNd, dim=2)
    Conv3d = partial(ConvNd, dim=3)

    UpSampling = staticmethod(UpSamplingNd)
    UpSampling1d = partial(UpSamplingNd, dim=1)
    UpSampling2d = partial(UpSamplingNd, dim=2)
    UpSampling3d = partial(UpSamplingNd, dim=3)

    ConvTransposed = partial(ConvNd, transposed=True)
    ConvTransposed1d = partial(ConvNd, transposed=True, dim=1)
    ConvTransposed2d = partial(ConvNd, transposed=True, dim=2)
    ConvTransposed3d = partial(ConvNd, transposed=True, dim=3)

    AvgPool = staticmethod(AvgPoolNd)
    AvgPool1d = partial(AvgPoolNd, dim=1)
    AvgPool2d = partial(AvgPoolNd, dim=2)
    AvgPool3d = partial(AvgPoolNd, dim=3)

    LayerNorm = staticmethod(LayerNorm)

    @staticmethod
    def Parameter(x, dtype=None, learnable=True):
        """A tracked parameter.

        Args:
            x (tensor): Initial value of the parameter.
            dtype (dtype, optional): Data type.
            learnable (bool, optional): Whether the parameter is learnable.

        Returns:
            :class:`tf.Variable`: Parameter.
        """
        dtype = dtype or tf.float32
        dtype = convert(dtype, B.TFDType)
        return tf.Variable(x, dtype=dtype, trainable=learnable)


interface = Interface()  #: The TensorFlow interface.


class Module(tf.keras.Model):
    """A TensorFlow module."""

    def __init__(self):
        super().__init__()
        self.nn = interface
        self.nps = nps.tensorflow
