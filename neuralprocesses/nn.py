from typing import Tuple
import math

import lab as B
from plum import Dispatcher

from .parallel import Parallel
from .util import register_module

__all__ = ["MLP", "UNet", "ConvNet", "Splitter"]


_dispatch = Dispatcher()


@register_module
class MLP:
    def __init__(
        self,
        dim_in: int,
        dim_hidden: int,
        dim_out: int,
        num_layers: int,
        nonlinearity=None,
        dtype=None,
    ):
        # Default to ReLUs.
        if nonlinearity is None:
            nonlinearity = self.nn.ReLU()

        # Build layers.
        if num_layers == 1:
            layers = self.nn.Linear(dim_in, dim_out, dtype=dtype)
        else:
            layers = [self.nn.Linear(dim_in, dim_hidden, dtype=dtype)]
            for _ in range(num_layers - 2):
                layers.append(nonlinearity)
                layers.append(self.nn.Linear(dim_hidden, dim_hidden, dtype=dtype))
            layers.append(nonlinearity)
            layers.append(self.nn.Linear(dim_hidden, dim_out, dtype=dtype))
        self.net = self.nn.Sequential(*layers)

    def __call__(self, x):
        x = B.transpose(x)
        x = self.net(x)
        x = B.transpose(x)
        return x


@register_module
class UNet:
    def __init__(
        self,
        dim: int,
        in_channels: int,
        out_channels: int,
        channels: Tuple[int, ...] = (8, 16, 16, 32, 32, 64),
        dtype=None,
    ):
        self.activation = self.nn.ReLU()
        self.num_halving_layers = len(channels)

        Conv = getattr(self.nn, f"Conv{dim}d")
        ConvTranspose = getattr(self.nn, f"ConvTransposed{dim}d")

        # First linear layer:
        self.initial_linear = Conv(
            in_channels=in_channels,
            out_channels=channels[0],
            kernel=1,
            dtype=dtype,
        )

        # Final linear layer:
        self.final_linear = Conv(
            in_channels=channels[0],
            out_channels=out_channels,
            kernel=1,
            dtype=dtype,
        )

        # Before turn layers:
        kernel = 5
        self.before_turn_layers = self.nn.ModuleList(
            [
                Conv(
                    in_channels=channels[max(i - 1, 0)],
                    out_channels=channels[i],
                    kernel=kernel,
                    stride=2,
                    dtype=dtype,
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

        self.after_turn_layers = self.nn.ModuleList(
            [
                ConvTranspose(
                    in_channels=get_num_in_channels(i),
                    out_channels=channels[max(i - 1, 0)],
                    kernel=kernel,
                    stride=2,
                    output_padding=1,
                    dtype=dtype,
                )
                for i in range(len(channels))
            ]
        )

    def __call__(self, x):
        h = self.initial_linear(x)

        hs = [self.activation(self.before_turn_layers[0](h))]
        for layer in self.before_turn_layers[1:]:
            hs.append(self.activation(layer(hs[-1])))

        # Now make the turn!

        h = self.activation(self.after_turn_layers[-1](hs[-1]))
        for h_prev, layer in zip(
            reversed(hs[:-1]), reversed(self.after_turn_layers[:-1])
        ):
            h = self.activation(layer(B.concat(h_prev, h, axis=1)))

        return self.final_linear(h)


@register_module
class ConvNet:
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
        activation = self.nn.ReLU()
        # To make it a drop-in substitute for :class:`UNet`.
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
                    ),
                    Conv(
                        in_channels=channels,
                        out_channels=channels,
                        kernel=1,
                    ),
                    activation,
                ]
            )
        layers.append(
            Conv(
                in_channels=in_channels,
                out_channels=out_channels,
                kernel=1,
            )
        )
        self.conv_net = self.nn.Sequential(*layers)

    def __call__(self, z):
        return self.conv_net(z)


@register_module
class Splitter:
    def __init__(self, *sizes):
        self.sizes = sizes

    @_dispatch
    def __call__(self, z: B.Numeric):
        i = 0
        splits = []
        for size in self.sizes:
            splits.append(z[:, i : i + size, :])
            i += size
        return Parallel(*splits)
