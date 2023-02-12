from typing import Union

import torch
from torch import nn

from .kernels import Kernel


class SparseGP(nn.Module):
    def __init__(
        self,
        kernel: Kernel,
        likelihood: nn.Module,
        init_z: torch.Tensor,
        mean_init_std: float = 1e-3,
        jitter: float = 1e-6,
    ):
        super().__init__()

        assert len(init_z.shape) == 2

        self.kernel = kernel
        self.likelihood = likelihood
        self.z = nn.Parameter(init_z)
        self.jitter = jitter

        # Initialise variaitonal parameters (in whitened space).
        self.qw_loc = nn.Parameter(torch.randn(self.num_inducing) * mean_init_std)
        chol_init = torch.eye(self.num_inducing)
        tril_indices = torch.tril_indices(self.num_inducing, self.num_inducing)

        self.qw_chol_diag = nn.Parameter((chol_init.diag().exp() - 1).log())
        self.qw_chol_other = nn.Parameter(
            chol_init[tril_indices[0], tril_indices[1]].flatten()
        )

    @property
    def num_inducing(self):
        return self.z.shape[0]

    @property
    def qw_chol(self):
        tril_indices = torch.tril_indices(self.num_inducing, self.num_inducing)
        qw_chol_other = torch.zeros((self.num_inducing, self.num_inducing))
        qw_chol_other[tril_indices[0], tril_indices[1]] = self.qw_chol_other

        qw_chol = torch.tril(
            torch.ones(self.num_inducing, self.num_inducing), diagonal=-1
        ) * qw_chol_other + (
            torch.diag_embed(nn.functional.softplus(self.qw_chol_diag))
        )

        return qw_chol

    @property
    def qw_cov(self):
        return self.qw_chol @ self.qw_chol.transpose(-1, -2)

    @property
    def qw(self):
        return torch.distributions.MultivariateNormal(
            self.qw_loc, scale_tril=self.qw_chol
        )

    @property
    def qu_loc(self):
        kzz = self.kernel(self.z, self.z) + torch.eye(self.num_inducing) * self.jitter
        lzz = torch.linalg.cholesky(kzz)

        return lzz @ self.qw_loc

    @property
    def qu_chol(self):
        kzz = self.kernel(self.z, self.z) + torch.eye(self.num_inducing) * self.jitter
        lzz = torch.linalg.cholesky(kzz)

        return lzz @ self.qw_chol

    @property
    def qu(self):
        kzz = self.kernel(self.z, self.z) + torch.eye(self.num_inducing) * self.jitter
        lzz = torch.linalg.cholesky(kzz)
        qu_loc = lzz @ self.qw_loc
        qu_chol = lzz @ self.qw_chol

        return torch.distributions.MultivariateNormal(qu_loc, scale_tril=qu_chol)

    def kl_divergence(self):
        kzz = self.kernel(self.z, self.z) + torch.eye(self.num_inducing) * self.jitter
        lzz = torch.linalg.cholesky(kzz)

        qu_loc = lzz @ self.qw_loc
        qu_chol = lzz @ self.qw_chol

        qu = torch.distributions.MultivariateNormal(qu_loc, scale_tril=qu_chol)

        pu_loc = torch.zeros_like(qu_loc)
        pu = torch.distributions.MultivariateNormal(pu_loc, scale_tril=lzz)

        return torch.distributions.kl_divergence(qu, pu)

    def forward(
        self, x: torch.Tensor
    ) -> Union[torch.distributions.Normal, torch.distributions.MultivariateNormal]:
        assert len(x.shape) == 2
        assert x.shape[-1] == self.z.shape[-1]

        kxx = self.kernel(x, x, diag=True)
        kxz = self.kernel(x, self.z)
        kzx = kxz.transpose(-1, -2)
        kzz = self.kernel(self.z, self.z) + torch.eye(self.num_inducing) * self.jitter

        # Compute interpolation terms.
        # K_ZZ^{-1/2} K_ZX
        # K_ZZ^{-1/2} \mu_Z
        lzz = torch.linalg.cholesky(kzz)
        interp_term = torch.linalg.solve(lzz, kzx)

        # Compute the mean of q(f).
        # k_XZ K_ZZ^{-1/2} (m - K_ZZ^{-1/2} \mu_Z) + \mu_X
        qf_loc = (interp_term.transpose(-1, -2) @ self.qw_loc.unsqueeze(-1)).squeeze(-1)

        # Compute the variance of q(f).
        middle_term = torch.eye(self.num_inducing) - self.qw_cov

        interp_term_ = interp_term.transpose(-1, -2).unsqueeze(-1)
        middle_term_ = middle_term.unsqueeze(0)
        qf_cov = (kxx + self.jitter) - (
            interp_term_.transpose(-1, -2) @ middle_term_ @ interp_term_
        ).squeeze(-1).squeeze(-1)

        return torch.stack((qf_loc, qf_cov), dim=-1)
