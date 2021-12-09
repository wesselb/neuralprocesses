import lab as B
from matrix import Diagonal
from stheno import Normal

__all__ = ["HeterogeneousGaussianLikelihood"]


class HeterogeneousGaussianLikelihood:
    def __call__(self, z):
        i = B.shape(z, 1)
        if i % 2 != 0:
            raise ValueError("Must give an even number of channels.")
        if i != 2:
            raise NotImplementedError("Multi-dim outputs not yet supported.")
        mu = z[:, : (i // 2), :]
        var = B.softplus(z[:, (i // 2) :, :])
        # Assume that the outputs are one-dimensional.
        return Normal(B.transpose(mu), Diagonal(B.transpose(var)[:, :, 0]))
