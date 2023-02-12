from typing import Optional

import torch
from torch import nn


def sq_dist(x1, x2):
    adjustment = x1.mean(-2, keepdim=True)
    x1 = x1 - adjustment

    # Compute squared distance matrix using quadratic expansion
    x1_norm = x1.pow(2).sum(dim=-1, keepdim=True)
    x1_pad = torch.ones_like(x1_norm)
    x2 = (
        x2 - adjustment
    )  # x1 and x2 should be identical in all dims except -2 at this point
    x2_norm = x2.pow(2).sum(dim=-1, keepdim=True)
    x2_pad = torch.ones_like(x2_norm)

    x1_ = torch.cat([-2.0 * x1, x1_norm, x1_pad], dim=-1)
    x2_ = torch.cat([x2, x2_pad, x2_norm], dim=-1)
    res = x1_.matmul(x2_.transpose(-2, -1))

    # Zero out negative values
    return res.clamp_min_(0)


def dist(x1, x2):
    res = sq_dist(x1, x2)
    return res.clamp_min_(1e-30).sqrt_()


class Kernel(nn.Module):
    def __init__(self, ard_num_dims: Optional[int] = None):
        super().__init__()

        self.ard_num_dims = ard_num_dims

        lengthscale_num_dims = 1 if ard_num_dims is None else ard_num_dims
        self.log_lengthscale = nn.Parameter(torch.zeros(1, lengthscale_num_dims))
        self.log_scale = nn.Parameter(torch.zeros(1))

    @property
    def lengthscale(self):
        return self.log_lengthscale.exp()

    @lengthscale.setter
    def lenthscale(self, value: float):
        lengthscale_num_dims = 1 if self.ard_num_dims is None else self.ard_num_dims
        self.log_lengthscale = nn.Parameter(
            (torch.ones(1, lengthscale_num_dims) * value).log()
        )

    @property
    def scale(self):
        return self.scale.exp()

    @scale.setter
    def scale(self, value: float):
        self.log_scale = nn.Parameter(torch.as_tensor(value).log())

    def forward(
        self, x1: torch.Tensor, x2: torch.Tensor, diag: bool = False
    ) -> torch.Tensor:
        raise NotImplementedError

    def covar_dist(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        square_dist: bool = False,
    ):
        res = None

        if diag:
            res = torch.linalg.norm(x1 - x2, dim=-1)  # 2-norm by default
            return res.pow(2) if square_dist else res
        dist_func = sq_dist if square_dist else dist
        return dist_func(x1, x2)

    def __call__(
        self,
        x1: torch.Tensor,
        x2: Optional[torch.Tensor] = None,
        diag: bool = False,
        **params,
    ) -> torch.Tensor:
        x1_, x2_ = x1, x2

        # Give x1_ and x2_ a last dimension, if necessary
        if x1_.ndimension() == 1:
            x1_ = x1_.unsqueeze(1)
        if x2_ is not None:
            if x2_.ndimension() == 1:
                x2_ = x2_.unsqueeze(1)
            if not x1_.size(-1) == x2_.size(-1):
                raise RuntimeError(
                    "x1_ and x2_ must have the same number of dimensions!"
                )

        if x2_ is None:
            x2_ = x1_

        return self.forward(x1_, x2_, diag=diag, **params)


class RBFKernel(Kernel):
    def forward(
        self,
        x1: torch.Tensor,
        x2: torch.Tensor,
        diag: bool = False,
        **params,
    ):
        x1_ = x1.div(self.lengthscale)
        x2_ = x2.div(self.lengthscale)
        return (
            self.covar_dist(x1_, x2_, square_dist=True, diag=diag, **params)
            .div(-2)
            .exp()
        )
