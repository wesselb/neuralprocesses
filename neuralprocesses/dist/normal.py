import lab as B
import numpy as np
from matrix import AbstractMatrix, Dense, Diagonal, LowRank, Woodbury
from plum import parametric, Union
from stheno import Normal
from wbml.util import indented_kv

from .dist import AbstractMultiOutputDistribution
from .. import _dispatch
from ..aggregate import Aggregate
from ..util import batch, split

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
def _monormal_unvectorise(x: B.Numeric, shape, *, squeeze_sample_dim=False):
    x = B.reshape(x, *B.shape(x)[:-1], *shape)
    if squeeze_sample_dim:
        x = B.squeeze(x, axis=0)
    return x


@_dispatch
def _monormal_unvectorise(x: B.Numeric, shape: Aggregate, **kw_args):
    ns = [np.prod(si) for si in shape]
    xs = split(x, ns, -1)
    return Aggregate(
        *(_monormal_unvectorise(xi, si, **kw_args) for xi, si in zip(xs, shape))
    )


@parametric
class MultiOutputNormal(AbstractMultiOutputDistribution):
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
        return self.vectorised_normal.logpdf(_monormal_vectorise(x, self.shape))

    @_dispatch
    def sample(self, state: B.RandomState, num: Union[B.Int, None] = None):
        def f(sample):
            # Put the sample dimension first.
            perm = list(range(B.rank(sample)))
            perm = perm[-1:] + perm[:-1]
            sample = B.transpose(sample, perm=perm)
            # Undo the vectorisation.
            sample = _monormal_unvectorise(
                sample,
                self.shape,
                # Squeeze the sample dimension if no number of samples was specified.
                squeeze_sample_dim=num is None,
            )
            return sample

        return _map_sample_output(f, self.vectorised_normal.sample(state, num or 1))

    @_dispatch
    def sample(self, num: Union[B.Int, None] = None):
        state = B.global_random_state(B.dtype(self._mean, self._var, self._noise))
        state, sample = self.sample(state, num)
        B.set_global_random_state(state)
        return sample

    def kl(self, other: "MultiOutputNormal"):
        return self.vectorised_normal.kl(other.vectorised_normal)

    def entropy(self):
        return self.vectorised_normal.entropy()


@B.dispatch
def dtype(dist: MultiOutputNormal):
    return B.dtype(dist.vectorised_normal)


@B.dispatch
def cast(dtype: B.DType, dist: MultiOutputNormal):
    return MultiOutputNormal(
        B.cast(dtype, dist._mean),
        B.cast(dtype, dist._var),
        B.cast(dtype, dist._noise),
        dist.shape,
    )


@B.dispatch
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
