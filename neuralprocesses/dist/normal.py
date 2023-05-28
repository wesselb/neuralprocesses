from typing import Union

import lab as B
import numpy as np
from matrix import AbstractMatrix, Dense, Diagonal, LowRank, Woodbury
from plum import parametric
from stheno import Normal
from wbml.util import indented_kv

from .. import _dispatch
from ..aggregate import Aggregate
from ..util import batch, split
from .dist import AbstractDistribution, shape_batch

__all__ = ["MultiOutputNormal"]


@_dispatch
def _map_sample_output(f, state_res: tuple):
    state, res = state_res
    return state, f(res)


@_dispatch
def _map_sample_output(f, res):
    return f(res)


@_dispatch
def _monormal_vectorise(x: B.Numeric, shape):
    return B.reshape(x, *batch(x, len(shape)), -1, 1)


@_dispatch
def _monormal_vectorise(x: Aggregate, shape: Aggregate):
    return B.concat(*(_monormal_vectorise(xi, si) for xi, si in zip(x, shape)), axis=-2)


@_dispatch
def _monormal_unvectorise(x: B.Numeric, data_shape, *, squeeze_sample_dim=False):
    x = B.reshape(x, *B.shape(x)[:-1], *data_shape)
    if squeeze_sample_dim:
        x = B.squeeze(x, axis=0)
    return x


@_dispatch
def _monormal_unvectorise(x: B.Numeric, data_shape: Aggregate, **kw_args):
    ns = [np.prod(si) for si in data_shape]
    xs = split(x, ns, -1)
    return Aggregate(
        *(_monormal_unvectorise(xi, di, **kw_args) for xi, di in zip(xs, data_shape))
    )


@parametric
class MultiOutputNormal(AbstractDistribution):
    """A normal distribution for multiple outputs. Use one of the class methods to
    construct the object.

    Args:
        mean (matrix): Mean of the underlying vectorised multivariate normal.
        var (matrix): Variance of the underlying vectorised multivariate normal.
        noise (matrix): Noise of the underlying vectorised multivariate normal.
        shape (shape or :class:`neuralprocesses.aggregate.Aggregate`): Shape(s) of the
            data before vectorising.

    Attributes:
        shape (shape or :class:`neuralprocesses.aggregate.Aggregate`): Shape(s) of the
            data before vectorising.
    """

    @_dispatch
    def __init__(
        self,
        mean: AbstractMatrix,
        var: AbstractMatrix,
        noise: AbstractMatrix,
        shape,
    ):
        self._mean = mean
        self._var = var
        self._noise = noise
        self.shape = shape

    @classmethod
    def __infer_type_parameter__(cls, mean, var, noise, shape):
        return type(mean), type(var), type(noise)

    @property
    def vectorised_normal(self):
        """:class:`stheno.Normal`: Underlying vectorised multivariate normal."""
        return Normal(self._mean, _possibly_densify_variance(self._var + self._noise))

    @property
    def noiseless(self):
        """:class:`.MultiOutputNormal`: Noiseless version of the distribution."""
        return MultiOutputNormal(
            self._mean,
            self._var,
            B.zeros(self._noise),
            self.shape,
        )

    @classmethod
    def dense(cls, mean: B.Numeric, var: B.Numeric, noise: B.Numeric, shape):
        """Construct a dense multi-output normal distribution.

        Args:
            mean (tensor): Mean of shape `(*b, n)`.
            var (tensor): Variance matrix of shape `(*b, n, n)`.
            noise (tensor): Marginal variances of shape `(*b, n)`.
            shape (shape): Shape of the data before vectorising.
        """
        return cls(Dense(mean[..., None]), Dense(var), Diagonal(noise), shape)

    @classmethod
    def diagonal(cls, mean: B.Numeric, noise: B.Numeric, shape):
        """Construct a diagonal multi-output normal distribution.

        Args:
            mean (tensor): Mean of shape `(*b, n)`.
            noise (tensor): Marginal variances of shape `(*b, n)`.
            shape (shape): Shape of the data before vectorising.
        """
        noise = Diagonal(noise)
        return cls(Dense(mean[..., None]), B.zeros(noise), noise, shape)

    @classmethod
    def lowrank(
        cls,
        mean: B.Numeric,
        var_factor: B.Numeric,
        noise: B.Numeric,
        shape,
    ):
        """Construct a low-rank multi-output normal distribution.

        Args:
            mean (tensor): Mean of shape `(*b, n)`.
            var_factor (tensor): Factors of the low-rank variance of shape `(*b, n, f)`.
            noise (tensor): Diagonal part of the low-rank variance of shape `(*b, n)`.
            shape (shape): Shape of the data before vectorising.
        """
        return cls(Dense(mean[..., None]), LowRank(var_factor), Diagonal(noise), shape)

    def __repr__(self):
        return (  # Comment to preserve formatting.
            f"<MultiOutputNormal: shape={self.shape}\n"
            + indented_kv("vectorised_normal", repr(self.vectorised_normal), suffix=">")
        )

    def __str__(self):
        return (  # Comment to preserve formatting.
            f"<MultiOutputNormal: shape={self.shape}\n"
            + indented_kv("vectorised_normal", str(self.vectorised_normal), suffix=">")
        )

    @property
    def mean(self):
        return _monormal_unvectorise(self.vectorised_normal.mean[..., 0], self.shape)

    @property
    def var(self):
        return _monormal_unvectorise(B.diag(self.vectorised_normal.var), self.shape)

    def logpdf(self, x):
        x = _monormal_vectorise(x, self.shape)
        d = self.vectorised_normal
        if B.jit_to_numpy(B.all(~B.isnan(x))):
            return d.logpdf(x)
        else:
            # Data is missing. Unfortunately, which elements are missing can differ
            # between batches. The only thing we can now do is to loop over batches.
            # For now, we only support a single batch dimension.
            if B.rank(x) > 3:
                raise NotImplementedError(
                    "`MultiOutputNormal` for now only supports missing data with "
                    "a single batch dimension."
                )
            logpdfs = []
            for b in range(B.shape(x, 0)):
                x_b = x[b]
                mask = ~B.isnan(x_b[:, 0])
                # Select the batch from the vectorised distribution.
                d_b = Normal(_index_batch(d.mean, b), _index_batch(d.var, b))
                # Select the non-missing elements.
                d_b = Normal(
                    B.take(d_b.mean, mask, axis=-2),
                    B.submatrix(d_b.var, mask),
                )
                logpdfs.append(d_b.logpdf(x_b[mask, :]))
            return B.stack(*logpdfs, axis=-1)

    @_dispatch
    def sample(self, state: B.RandomState, dtype: B.DType, *shape: B.Int):
        def f(sample):
            # Put the sample dimension first.
            perm = list(range(B.rank(sample)))
            perm = perm[-1:] + perm[:-1]
            sample = B.transpose(sample, perm=perm)
            # Undo the vectorisation.
            return _monormal_unvectorise(
                sample,
                self.shape,
                # If `shape` is not specified, then the sample dimension of
                # :class:`stheno.Normal` needs to be squeezed.
                squeeze_sample_dim=shape == (),
            )

        # TODO: Use `dtype` here. :class:`stheno.Normal` doesn't yet support `dtype`.
        return _map_sample_output(f, self.vectorised_normal.sample(state, *shape))

    def kl(self, other: "MultiOutputNormal"):
        return self.vectorised_normal.kl(other.vectorised_normal)

    def entropy(self):
        return self.vectorised_normal.entropy()


@B.dispatch
def dtype(dist: MultiOutputNormal):
    return B.dtype(dist._mean, dist._var, dist._noise)


@B.dispatch
def cast(dtype: B.DType, dist: MultiOutputNormal):
    return MultiOutputNormal(
        B.cast(dtype, dist._mean),
        B.cast(dtype, dist._var),
        B.cast(dtype, dist._noise),
        dist.shape,
    )


@shape_batch.dispatch
def shape_batch(dist: MultiOutputNormal):
    return B.shape_batch_broadcast(dist._mean, dist._var, dist._noise)


@_dispatch
def _possibly_densify_variance(var: AbstractMatrix):
    return var


@_dispatch
def _possibly_densify_variance(var: Woodbury):
    if var.lr.rank >= B.shape_matrix(var, 0):
        return B.dense(var)
    else:
        return var


@_dispatch
def _index_batch(a: B.Numeric, i: int):
    return a[..., i, :, :]


@_dispatch
def _index_batch(a: Dense, i: int):
    return Dense(a.mat[..., i, :, :])


@_dispatch
def _index_batch(a: Diagonal, i: int):
    return Diagonal(a.diag[..., i, :])


@_dispatch
def _index_batch(a: LowRank, i: int):
    return LowRank(
        left=_index_batch(a.left, i),
        right=_index_batch(a._right, i) if a._right else None,
        middle=_index_batch(a._middle, i) if a._middle else None,
    )


@_dispatch
def _index_batch(a: Woodbury, i: int):
    return _index_batch(a.diag, i) + _index_batch(a.lr, i)
