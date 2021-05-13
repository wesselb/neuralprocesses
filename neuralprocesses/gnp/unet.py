from typing import Tuple

import lab.torch as B
import torch.nn as nn
from plum import Dispatcher

_dispatch = Dispatcher()


@_dispatch
def init(x: nn.Module):
    nn.init.xavier_normal_(x.weight, gain=1)
    nn.init.constant_(x.bias, 1e-3)


@_dispatch
def init(xs: nn.Sequential):
    for x in xs:
        init(x)


class UNet(nn.Module):
    def __init__(
        self,
        dimensionality: int,
        in_channels: int,
        out_channels: int,
        channels: Tuple[int, ...] = (8, 16, 16, 32, 32, 64),
    ):

        super(UNet, self).__init__()

        self.activation = nn.ReLU()
        self.num_halving_layers = len(channels)

        Conv = getattr(nn, f"Conv{dimensionality}d")
        ConvTranspose = getattr(nn, f"ConvTranspose{dimensionality}d")

        # First linear layer:
        self.initial_linear = Conv(
            in_channels=in_channels,
            out_channels=channels[0],
            kernel_size=1,
            padding=0,
            stride=1,
        )
        init(self.initial_linear)

        # Final linear layer:
        self.final_linear = Conv(
            in_channels=channels[0],
            out_channels=out_channels,
            kernel_size=1,
            padding=0,
            stride=1,
        )
        init(self.final_linear)

        # Before turn layers:
        kernel_size = 5
        padding = kernel_size // 2
        self.before_turn_layers = nn.Sequential(
            *[
                Conv(
                    in_channels=channels[max(i - 1, 0)],
                    out_channels=channels[i],
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=2,
                )
                for i in range(len(channels))
            ]
        )
        init(self.before_turn_layers)

        # After turn layers:

        def get_num_in_channels(i):
            if i == len(channels) - 1:
                # No skip connection yet.
                return channels[i]
            else:
                # Add the skip connection.
                return 2 * channels[i]

        self.after_turn_layers = nn.Sequential(
            *[
                ConvTranspose(
                    in_channels=get_num_in_channels(i),
                    out_channels=channels[max(i - 1, 0)],
                    kernel_size=kernel_size,
                    padding=padding,
                    stride=2,
                    output_padding=1,
                )
                for i in range(len(channels))
            ]
        )
        init(self.after_turn_layers)

    def forward(self, x):
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
