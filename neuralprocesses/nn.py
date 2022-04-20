import math
from typing import Tuple, Union

import lab as B
import numpy as np
from plum import Dispatcher

from .parallel import Parallel
from .util import register_module, batch, compress_batch_dimensions

__all__ = ["MLP", "UNet", "ConvNet", "Splitter"]


_dispatch = Dispatcher()


@register_module
class MLP:
    """MLP.

    Args:
        in_dim (int): Input dimensionality.
        layers (tuple[int, ...]): Width of every hidden layer.
        out_dim (int): Output dimensionality.
        nonlinearity (function, optional): Nonlinearity.
        dtype (dtype, optional): Data type.

    Attributes:
        net (object): MLP, but which expects a different data format.
    """

    def __init__(
        self,
        in_dim: int,
        layers: Tuple[int, ...],
        out_dim: int,
        nonlinearity=None,
        dtype=None,
    ):

        # Default to ReLUs.
        if nonlinearity is None:
            nonlinearity = self.nn.ReLU()

        # Build layers.
        if len(layers) == 0:
            self.net = self.nn.Linear(in_dim, out_dim, dtype=dtype)
        else:
            net = [self.nn.Linear(in_dim, layers[0], dtype=dtype)]
            for i in range(1, len(layers)):
                net.append(nonlinearity)
                net.append(self.nn.Linear(layers[i - 1], layers[i], dtype=dtype))
            net.append(nonlinearity)
            net.append(self.nn.Linear(layers[-1], out_dim, dtype=dtype))
            self.net = self.nn.Sequential(*net)

    def __call__(self, x):
        x, uncompress = compress_batch_dimensions(x, 2)
        x = B.transpose(x)
        x = self.net(x)
        x = B.transpose(x)
        return uncompress(x)


@register_module
class UNet:
    """UNet.

    Args:
        dim (int): Dimensionality.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channels (tuple[int], optional): Channels of every layer of the UNet.
            Defaults to six layers each with 64 channels.
        unet_kernels (int or tuple[int], optional): Sizes of the kernels. Defaults to
            `5`.
        activations (object or tuple[object], optional): Activation functions.
        resize_convs (bool, optional): Use resize convolutions rather than
            transposed convolutions. Defaults to `False`.
        resize_conv_interp_method (str, optional): Interpolation method for the
            resize convolutions. Can be set to "bilinear". Defaults to "nearest".
        dtype (dtype, optional): Data type.

    Attributes:
        dim (int): Dimensionality.
        kernels (tuple[int]): Sizes of the kernels.
        activations (tuple[function]): Activation functions.
        num_halving_layers (int): Number of layers with striding equal to two.
        receptive_fields (list[float]): Receptive field for every intermediate value.
        receptive_field (float): Receptive field of the model.
        before_turn_layers (list[module]): Layers before the U-turn.
        after_turn_layers (list[module]): Layers after the U-turn
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        channels: Tuple[int, ...] = (8, 16, 16, 32, 32, 64),
        kernels: Union[int, Tuple[int, ...]] = 5,
        activations: Union[None, object, Tuple[object, ...]] = None,
        resize_convs: bool = False,
        resize_conv_interp_method: str = "nearest",
        dtype=None,
    ):
        self.dim = dim

        # If `kernel` is an integer, repeat it for every layer.
        if not isinstance(kernels, (tuple, list)):
            kernels = (kernels,) * len(channels)
        elif len(kernels) != len(channels):
            raise ValueError(
                f"Length of `kernels` ({len(kernels)}) must equal "
                f"the length of `channels` ({len(channels)})."
            )
        self.kernels = kernels

        # Default to ReLUs. Moreover, if `activations` is an activation function, repeat
        # it for every layer.
        activations = activations or self.nn.ReLU()
        if not isinstance(activations, (tuple, list)):
            activations = (activations,) * len(channels)
        elif len(activations) != len(channels):
            raise ValueError(
                f"Length of `activations` ({len(activations)}) must equal "
                f"the length of `channels` ({len(channels)})."
            )
        self.activations = activations

        # Compute number of halving layers.
        self.num_halving_layers = len(channels)

        # Compute receptive field at all stages of the model.
        self.receptive_fields = [1]
        # Forward pass:
        for kernel in self.kernels:
            after_conv = self.receptive_fields[-1] + (kernel - 1)
            if after_conv % 2 == 0:
                # If even, then subsample.
                self.receptive_fields.append(after_conv // 2)
            else:
                # If odd, then average pool.
                self.receptive_fields.append((after_conv + 1) // 2)
        # Backward pass:
        for kernel in reversed(self.kernels):
            after_interp = self.receptive_fields[-1] * 2 - 1
            self.receptive_fields.append(after_interp + (kernel - 1))
        self.receptive_field = self.receptive_fields[-1]

        Conv = getattr(self.nn, f"Conv{dim}d")
        ConvTranspose = getattr(self.nn, f"ConvTransposed{dim}d")
        UpSampling = getattr(self.nn, f"UpSampling{dim}d")
        AvgPool = getattr(self.nn, f"AvgPool{dim}d")

        # Final linear layer:
        self.final_linear = Conv(
            in_channels=channels[0],
            out_channels=out_channels,
            kernel=1,
            dtype=dtype,
        )

        # Before turn layers:
        self.before_turn_layers = self.nn.ModuleList(
            [
                (
                    # Perform average pooling if the previous receptive field is odd.
                    self.nn.Sequential(
                        Conv(
                            in_channels=((in_channels,) + channels)[i],
                            out_channels=channels[i],
                            kernel=self.kernels[i],
                            stride=1,
                            dtype=dtype,
                        ),
                        AvgPool(kernel=2, stride=2, dtype=dtype),
                    )
                    if self.receptive_fields[i] % 2 == 1
                    # Perform subsampling if the previous receptive field is even.
                    else Conv(
                        in_channels=((in_channels,) + channels)[i],
                        out_channels=channels[i],
                        kernel=self.kernels[i],
                        stride=2,
                        dtype=dtype,
                    )
                )
                for i in range(len(channels))
            ]
        )

        # After turn layers:

        def get_num_in_channels(i):
            if i == len(channels) - 1:
                # No skip connection yet.
                return channels[i]
            else:
                # Add the skip connection.
                return 2 * channels[i]

        def after_turn_layer(i):
            if resize_convs:
                return self.nn.Sequential(
                    UpSampling(
                        interp_method=resize_conv_interp_method,
                        dtype=dtype,
                    ),
                    Conv(
                        in_channels=get_num_in_channels(i),
                        out_channels=((channels[0],) + channels)[i],
                        kernel=self.kernels[i],
                        stride=1,
                        dtype=dtype,
                    ),
                )
            else:
                return ConvTranspose(
                    in_channels=get_num_in_channels(i),
                    out_channels=((channels[0],) + channels)[i],
                    kernel=self.kernels[i],
                    stride=2,
                    output_padding=1,
                    dtype=dtype,
                )

        self.after_turn_layers = self.nn.ModuleList(
            [after_turn_layer(i) for i in range(len(channels))]
        )

    def __call__(self, x):
        x, uncompress = compress_batch_dimensions(x, self.dim + 1)

        hs = [self.activations[0](self.before_turn_layers[0](x))]
        for layer, activation in zip(
            self.before_turn_layers[1:],
            self.activations[1:],
        ):
            hs.append(activation(layer(hs[-1])))

        # Now make the turn!

        h = self.activations[-1](self.after_turn_layers[-1](hs[-1]))
        for h_prev, layer, activation in zip(
            reversed(hs[:-1]),
            reversed(self.after_turn_layers[:-1]),
            reversed(self.activations[:-1]),
        ):
            h = activation(layer(B.concat(h_prev, h, axis=1)))

        return uncompress(self.final_linear(h))


@register_module
class ConvNet:
    """A depthwise-separable convolutional neural network.

    Args:
        dim (int): Dimensionality.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channels (int): Number of channels at every intermediate layer.
        num_layers (int): Number of layers.
        points_per_unit (float): Density of the discretisation corresponding to the
            inputs.
        receptive_field (float): Desired receptive field.
        dtype (dtype, optional): Data type.

    Attributes:
        dim (int): Dimensionality.
        num_halving_layers (int): Number of layers with stride equal to two.
        conv_net (module): The architecture.
    """

    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        channels: int,
        num_layers: int,
        points_per_unit: float,
        receptive_field: float,
        dtype=None,
    ):
        self.dim = dim

        activation = self.nn.ReLU()

        # Make it a drop-in substitute for :class:`UNet`.
        self.num_halving_layers = 0

        # Compute kernel size.
        receptive_points = receptive_field * points_per_unit
        kernel = math.ceil(1 + (receptive_points - 1) / num_layers)
        kernel = kernel + 1 if kernel % 2 == 0 else kernel  # Make kernel size odd.

        Conv = getattr(self.nn, f"Conv{dim}d")
        layers = [
            Conv(
                in_channels=in_channels,
                out_channels=channels,
                kernel=1,
                dtype=dtype,
            ),
            activation,
        ]
        for _ in range(num_layers):
            layers.extend(
                [
                    Conv(
                        in_channels=channels,
                        out_channels=channels,
                        kernel=kernel,
                        groups=channels,
                        dtype=dtype,
                    ),
                    Conv(
                        in_channels=channels,
                        out_channels=channels,
                        kernel=1,
                        dtype=dtype,
                    ),
                    activation,
                ]
            )
        layers.append(
            Conv(
                in_channels=channels,
                out_channels=out_channels,
                kernel=1,
                dtype=dtype,
            )
        )
        self.conv_net = self.nn.Sequential(*layers)

    def __call__(self, x):
        x, uncompress = compress_batch_dimensions(x, self.dim + 1)
        return uncompress(self.conv_net(x))


@register_module
class Splitter:
    """Split a tensor into multiple tensors.

    Args:
        *sizes (int): Size of every split

    Attributes:
        sizes (tuple[int]): Size of every split
    """

    def __init__(self, *sizes):
        self.sizes = sizes

    @_dispatch
    def __call__(self, z: B.Numeric):
        i = 0
        splits = []
        for size in self.sizes:
            splits.append(z[..., i : i + size, :])
            i += size
        return Parallel(*splits)
