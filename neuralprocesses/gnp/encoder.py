import lab.torch as B
import torch
import torch.nn as nn


__all__ = ["SetConv1dEncoder", "SetConv2dEncoder"]


class SetConv1dEncoder(nn.Module):
    def __init__(self, discretisation):
        nn.Module.__init__(self)
        self.log_scale = nn.Parameter(
            B.log(torch.tensor(2 / discretisation.points_per_unit)),
            requires_grad=True,
        )
        self.discretisation = discretisation

    def forward(self, xz, z, x):
        with B.device(B.device(z)):
            # Construct grid and density.
            x_grid = self.discretisation(xz, x)
            density_channel = B.ones(B.dtype(z), *B.shape(z)[:2], 1)

        # Prepend density channel.
        z = B.concat(density_channel, z, axis=2)

        # Compute interpolation weights.
        dists2 = B.pw_dists2(x_grid[None, :], xz)
        weights = B.exp(-0.5 * dists2 / B.exp(2 * self.log_scale))

        # Interpolate to grid.
        z = B.matmul(weights, z)

        # Normalise by density channel.
        z = B.concat(z[:, :, :1], z[:, :, 1:] / (z[:, :, :1] + 1e-8), axis=2)

        # Put feature channel second.
        z = B.transpose(z)

        return x_grid, z


class SetConv2dEncoder(nn.Module):
    def __init__(self, discretisation):
        nn.Module.__init__(self)
        self.log_scale = nn.Parameter(
            B.log(torch.tensor(2 / discretisation.points_per_unit)),
            requires_grad=True,
        )
        self.discretisation = discretisation

    def forward(self, xz, z, x):
        with B.device(B.device(z)):
            # Construct grid, density, identity channel.
            x_grid = self.discretisation(xz, x)
            density_channel = B.ones(B.dtype(z), *B.shape(z)[:2], 1)
            identity_channel = B.eye(
                B.dtype(z),
                B.shape(z)[0],
                1,
                B.shape(x_grid)[0],
                B.shape(x_grid)[0],
            )

        # Prepend density channel.
        z = B.concat(density_channel, z, axis=2)

        # Put feature/channel dimension second and make a four-tensor.
        z = B.transpose(z)[..., None]

        # Compute interpolation weights.
        dists2 = B.pw_dists2(xz, x_grid[None, :])
        weights = B.exp(-0.5 * dists2 / B.exp(2 * self.log_scale))
        weights = weights[:, None, :, :]  # Insert channel dimension.

        # Interpolate to grid.
        z = B.matmul(weights * z, weights, tr_a=True)

        # Normalise by density channel.
        z = B.concat(z[:, :1, ...], z[:, 1:, ...] / (z[:, :1, ...] + 1e-8), axis=1)

        # Prepend identity channel to complete the encoding.
        z = B.concat(identity_channel, z, axis=1)

        return x_grid, z
