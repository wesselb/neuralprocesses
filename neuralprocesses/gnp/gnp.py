import lab.torch as B
import torch
import torch.nn as nn

from .decoder import SetConv1dDecoder, SetConv2dDecoder
from .discretisation import Discretisation1d
from .encoder import SetConv1dEncoder, SetConv2dEncoder
from .unet import UNet

__all__ = ["GNP"]


class GNP(nn.Module):
    def __init__(
        self,
        x_target=None,
        y_target_dim: int = 1,
        sigma: float = 0.1,
        points_per_unit: float = 32,
    ):
        super(GNP, self).__init__()

        self.x_target = x_target

        # Construct CNNs:
        self.conv_mean = UNet(
            dimensionality=1,
            in_channels=y_target_dim + 1,
            out_channels=y_target_dim,
        )
        self.conv_kernel = UNet(
            dimensionality=2,
            in_channels=y_target_dim + 2,
            out_channels=1,
        )

        # Construct discretisations:
        self.disc_mean = Discretisation1d(
            points_per_unit=points_per_unit,
            multiple=2 ** self.conv_mean.num_halving_layers,
            margin=0.1,
        )
        self.disc_kernel = Discretisation1d(
            points_per_unit=points_per_unit,
            multiple=2 ** self.conv_kernel.num_halving_layers,
            margin=0.1,
        )

        # Construct encoders:
        self.encoder_mean = SetConv1dEncoder(self.disc_mean)
        self.encoder_kernel = SetConv2dEncoder(self.disc_kernel)

        # Construct decoders:
        self.decoder_mean = SetConv1dDecoder(self.disc_mean)
        self.decoder_kernel = SetConv2dDecoder(self.disc_kernel)

        # Learnable observation noise:
        self.log_sigma = nn.Parameter(
            B.log(torch.tensor(sigma, dtype=torch.float32)),
            requires_grad=True,
        )

    def forward(self, x_context, y_context, x_target=None):
        if x_target is None:
            if self.x_target is None:
                raise ValueError("Must provide target inputs")
            else:
                x_target = self.x_target

        # Run mean architecture:
        xz, z = self.encoder_mean(x_context, y_context, x_target)
        z = self.conv_mean(z)
        mean = self.decoder_mean(xz, z, x_target)[1]

        if B.shape(mean)[2] != 1:
            raise NotImplementedError("Only one-dimensional outputs are supported.")
        mean = mean[:, :, 0]

        # Run kernel architecture:
        xz, z = self.encoder_kernel(x_context, y_context, x_target)
        z = self.conv_kernel(z)
        cov = self.decoder_kernel(xz, z, x_target)[1][:, 0, :, :]

        # Add observation noise.
        with B.device(B.device(cov)):
            cov = cov + B.eye(cov) * B.exp(self.log_sigma)

        return torch.distributions.MultivariateNormal(
            loc=mean,
            covariance_matrix=cov,
        )
