import math
from typing import Tuple, Union, Optional

import lab as B

from .. import _dispatch
from ..datadims import data_dims
from ..util import register_module, compress_batch_dimensions

__all__ = ["Linear", "MLP", "UNet", "ConvNet", "ResidualBlock"]


@register_module
class Linear:
    """A linear layer over channels.

    Args:
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        dtype (dtype, optional): Data type.

    Attributes:
        net (object): Linear layer.
    """

    def __init__(self, in_channels, out_channels, dtype):
        self.net = self.nn.Linear(in_channels, out_channels, dtype=dtype)


@register_module
class MLP:
    """MLP.

    Args:
        in_dim (int): Input dimensionality.
        out_dim (int): Output dimensionality.
        layers (tuple[int, ...], optional): Width of every hidden layer.
        num_layers (int, optional): Number of hidden layers.
        width (int, optional): Width of the hidden layers
        nonlinearity (function, optional): Nonlinearity.
        dtype (dtype, optional): Data type.

    Attributes:
        net (object): MLP, but which expects a different data format.
    """

    def __init__(
        self,
        in_dim: int,
        out_dim: int,
        layers: Optional[Tuple[int, ...]] = None,
        num_layers: Optional[int] = None,
        width: Optional[int] = None,
        nonlinearity=None,
        dtype=None,
    ):
        # Check that one of the two specifications is given.
        layers_given = layers is not None
        num_layers_given = num_layers is not None and width is not None
        if not (layers_given or num_layers_given):
            raise ValueError(
                f"Must specify either `layers` or `num_layers` and `width`."
            )
        # Make sure that `layers` is a tuple of various widths.
        if not layers_given and num_layers_given:
            layers = (width,) * num_layers

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
        x = B.transpose(x)
        x, uncompress = compress_batch_dimensions(x, 1)
        x = uncompress(self.net(x))
        x = B.transpose(x)
        return x


@_dispatch
def code(coder: Union[Linear, MLP], xz, z: B.Numeric, x, **kw_args):
    d = data_dims(xz)

    # Construct permutation to switch the channel dimension and the last dimension.
    switch = list(range(B.rank(z)))
    switch[-d - 1], switch[-1] = switch[-1], switch[-d - 1]

    # Switch, apply network after compressing the batch dimensions, and switch back.
    z = B.transpose(z, perm=switch)
    z, uncompress = compress_batch_dimensions(z, 1)
    z = uncompress(coder.net(z))
    z = B.transpose(z, perm=switch)

    return xz, z


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
        activation (object, optional): Activation functions.
        residual (bool, optional): Use residual blocks. Defaults to `False`.
        separable (bool, optional): Use depthwise separable convolutions. Defaults to
            `False`.
        resize_convs (bool, optional): Use resize convolutions rather than
            transposed convolutions. Defaults to `False`.
        resize_conv_interp_method (str, optional): Interpolation method for the
            resize convolutions. Can be set to "bilinear". Defaults to "nearest".
        dtype (dtype, optional): Data type.

    Attributes:
        dim (int): Dimensionality.
        kernels (tuple[int]): Sizes of the kernels.
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
        activation: Union[object, None] = None,
        residual: bool = False,
        separable: bool = False,
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

        # Default to ReLUs.
        activation = activation or self.nn.ReLU()

        # Compute number of halving layers.
        self.num_halving_layers = len(channels)

        # Compute receptive field at all stages of the model.
        self.receptive_fields = [1]
        # Forward pass:
        for kernel in self.kernels:
            # Deal with repeated convs:
            if isinstance(kernel, tuple):
                kernel = kernel[0] + sum([k - 1 for k in kernel[1:]])
            after_conv = self.receptive_fields[-1] + (kernel - 1)
            if after_conv % 2 == 0:
                # If even, then subsample.
                self.receptive_fields.append(after_conv // 2)
            else:
                # If odd, then average pool.
                self.receptive_fields.append((after_conv + 1) // 2)
        # Backward pass:
        for kernel in reversed(self.kernels):
            # Deal with repeated convs:
            if isinstance(kernel, tuple):
                kernel = kernel[0] + sum([k - 1 for k in kernel[1:]])
            after_interp = self.receptive_fields[-1] * 2 - 1
            self.receptive_fields.append(after_interp + (kernel - 1))
        self.receptive_field = self.receptive_fields[-1]

        UpSampling = getattr(self.nn, f"UpSampling{dim}d")
        AvgPool = getattr(self.nn, f"AvgPool{dim}d")

        # Final linear layer:
        self.final_linear = _conv_block(
            self,
            dim=dim,
            in_channels=channels[0],
            out_channels=out_channels,
            kernel=1,
            activation=activation,
            preactivation=True,
            dtype=dtype,
        )

        # Before turn layers:
        self.before_turn_layers = self.nn.ModuleList(
            [
                (
                    # Perform average pooling if the previous receptive field is odd.
                    self.nn.Sequential(
                        _conv_block(
                            self,
                            dim=dim,
                            in_channels=((in_channels,) + channels)[i],
                            out_channels=channels[i],
                            kernel=self.kernels[i],
                            stride=1,
                            activation=activation,
                            preactivation=i > 0,
                            separable=separable,
                            residual=residual,
                            dtype=dtype,
                        ),
                        AvgPool(kernel=2, stride=2, dtype=dtype),
                    )
                    if self.receptive_fields[i] % 2 == 1
                    # Perform subsampling if the previous receptive field is even.
                    else _conv_block(
                        self,
                        dim=dim,
                        in_channels=((in_channels,) + channels)[i],
                        out_channels=channels[i],
                        kernel=self.kernels[i],
                        stride=2,
                        activation=activation,
                        preactivation=i > 0,
                        separable=separable,
                        residual=residual,
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
                    _conv_block(
                        self,
                        dim=dim,
                        in_channels=get_num_in_channels(i),
                        out_channels=((channels[0],) + channels)[i],
                        kernel=self.kernels[i],
                        stride=1,
                        activation=activation,
                        preactivation=True,
                        residual=residual,
                        separable=separable,
                        dtype=dtype,
                    ),
                )
            else:
                return _conv_block(
                    self,
                    dim=dim,
                    in_channels=get_num_in_channels(i),
                    out_channels=((channels[0],) + channels)[i],
                    kernel=self.kernels[i],
                    stride=2,
                    activation=activation,
                    preactivation=True,
                    residual=residual,
                    separable=separable,
                    transposed=True,
                    dtype=dtype,
                )

        self.after_turn_layers = self.nn.ModuleList(
            [after_turn_layer(i) for i in range(len(channels))]
        )

    def __call__(self, x):
        x, uncompress = compress_batch_dimensions(x, self.dim + 1)

        hs = [self.before_turn_layers[0](x)]
        for layer in self.before_turn_layers[1:]:
            hs.append(layer(hs[-1]))

        # Now make the turn!

        h = self.after_turn_layers[-1](hs[-1])
        for h_prev, layer in zip(
            reversed(hs[:-1]),
            reversed(self.after_turn_layers[:-1]),
        ):
            h = layer(B.concat(h_prev, h, axis=1))

        return uncompress(self.final_linear(h))


@register_module
class ConvNet:
    """A regular convolutional neural network.

    Args:
        dim (int): Dimensionality.
        in_channels (int): Number of input channels.
        out_channels (int): Number of output channels.
        channels (int): Number of channels at every intermediate layer.
        num_layers (int): Number of layers.
        points_per_unit (float): Density of the discretisation corresponding to the
            inputs.
        receptive_field (float): Desired receptive field.
        separable (bool, optional): Use depthwise separable convolutions. Defaults
            to `True`.
        dtype (dtype, optional): Data type.

    Attributes:
        dim (int): Dimensionality.
        kernel (int): Kernel size.
        num_halving_layers (int): Number of layers with stride equal to two.
        receptive_field (float): Receptive field.
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
        separable: bool = True,
        residual: bool = False,
        dtype=None,
    ):
        self.dim = dim

        # Make it a drop-in substitute for :class:`UNet`.
        self.num_halving_layers = 0
        self.receptive_field = receptive_field

        # Compute kernel size.
        receptive_points = receptive_field * points_per_unit
        kernel = math.ceil(1 + (receptive_points - 1) / num_layers)
        kernel = kernel + 1 if kernel % 2 == 0 else kernel  # Make kernel size odd.
        self.kernel = kernel  # Store it for reference.

        # Construct the network.
        activation = self.nn.ReLU()
        self.conv_net = self.nn.Sequential(
            *(
                _conv_block(
                    self,
                    dim=dim,
                    in_channels=i,
                    out_channels=o,
                    kernel=kernel,
                    activation=activation,
                    preactivation=p,
                    separable=separable,
                    residual=residual,
                    dtype=dtype,
                )
                for i, o, p in zip(
                    (in_channels,) + (channels,) * (num_layers - 1),
                    (channels,) * (num_layers - 1) + (out_channels,),
                    (False,) + (True,) * (num_layers - 1),
                )
            )
        )

    def __call__(self, x):
        x, uncompress = compress_batch_dimensions(x, self.dim + 1)
        return uncompress(self.conv_net(x))


@register_module
class ResidualBlock:
    """Block of a residual network.

    Args:
        layer1 (object): First branch.
        layer2 (object): Second branch.
        layer_post (object): Layer after after the addition.

    Attributes:
        layer1 (object): First branch.
        layer2 (object): Second branch.
        layer_post (object): Linear layer after the addition.
    """

    def __init__(self, layer1, layer2, layer_post):
        self.layer1 = layer1
        self.layer2 = layer2
        self.layer_post = layer_post

    def __call__(self, x):
        return self.layer_post(self.layer1(x) + self.layer2(x))


def _conv_block(
    coder,
    *,
    dim,
    in_channels,
    out_channels,
    kernel,
    stride: int = 1,
    activation,
    preactivation: bool,
    separable=False,
    residual=False,
    transposed=False,
    dtype,
):
    channels = min(in_channels, out_channels)

    if residual:
        if in_channels == channels and stride == 1:
            # The input can be directly added to the output.
            input_transform = lambda x: x
        else:
            # The input cannot be directly added to the output, so use an additional
            # pointwise transform.
            input_transform = _conv(
                coder,
                dim=dim,
                in_channels=in_channels,
                out_channels=channels,
                kernel=1,
                stride=stride,
                transposed=transposed,
                separable=False,
                dtype=dtype,
            )
        return coder.nps.ResidualBlock(
            input_transform,
            coder.nn.Sequential(
                _conv_block(
                    coder,
                    dim=dim,
                    in_channels=in_channels,
                    out_channels=channels,
                    kernel=kernel,
                    stride=stride,
                    activation=activation,
                    preactivation=preactivation,
                    separable=separable,
                    residual=False,
                    transposed=transposed,
                    dtype=dtype,
                ),
                activation,
                _conv(
                    coder,
                    dim=dim,
                    in_channels=channels,
                    out_channels=channels,
                    kernel=1,
                    stride=1,
                    separable=False,
                    transposed=False,
                    dtype=dtype,
                ),
            ),
            _conv(
                coder,
                dim=dim,
                in_channels=channels,
                out_channels=out_channels,
                kernel=1,
                stride=1,
                separable=False,
                transposed=False,
                dtype=dtype,
            ),
        )

    if isinstance(kernel, tuple):
        nk = len(kernel)
        return coder.nn.Sequential(
            *(
                _conv_block(
                    coder,
                    dim=dim,
                    in_channels=i,
                    out_channels=o,
                    kernel=k,
                    stride=s,
                    activation=activation,
                    preactivation=p,
                    separable=separable,
                    residual=False,
                    transposed=t,
                    dtype=dtype,
                )
                for i, o, k, s, p, t in zip(
                    (in_channels,) + (channels,) * (nk - 1),
                    (channels,) * (nk - 1) + (out_channels,),
                    kernel,
                    (1,) * (nk - 1) + (stride,),
                    (preactivation,) + (True,) * (nk - 1),
                    (False,) * (nk - 1) + (transposed,),
                )
            )
        )

    if separable:
        return coder.nn.Sequential(
            *((activation,) if preactivation else ()),
            _conv(
                coder,
                dim=dim,
                in_channels=in_channels,
                out_channels=in_channels,
                kernel=kernel,
                stride=stride,
                separable=True,
                transposed=transposed,
                dtype=dtype,
            ),
            _conv(
                coder,
                dim=dim,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=1,
                stride=1,
                separable=False,
                transposed=False,
                dtype=dtype,
            ),
        )
    else:
        return coder.nn.Sequential(
            *((activation,) if preactivation else ()),
            _conv(
                coder,
                dim=dim,
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=kernel,
                stride=stride,
                separable=False,
                transposed=transposed,
                dtype=dtype,
            ),
        )


def _conv(
    coder,
    *,
    dim,
    kernel,
    stride,
    in_channels,
    out_channels,
    separable,
    transposed,
    dtype,
):
    output_padding = {}
    if transposed:
        Conv = getattr(coder.nn, f"ConvTransposed{dim}d")
        if stride > 1:
            output_padding = {"output_padding": stride // 2}
    else:
        Conv = getattr(coder.nn, f"Conv{dim}d")
    return Conv(
        in_channels=in_channels,
        out_channels=out_channels,
        kernel=kernel,
        stride=stride,
        **output_padding,
        groups=in_channels if separable else 1,
        dtype=dtype,
    )
