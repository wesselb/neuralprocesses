import lab as B
from functools import partial
from wbml.util import indented_kv

from .. import _dispatch

from .dist import AbstractMultiOutputDistribution
from .normal import _map_sample_output
from ..util import register_module
from ..aggregate import Aggregate
from ..datadims import data_dims

__all__ = ["Transform", "TransformedMultiOutputDistribution"]


@register_module
class Transform:
    """A transform for distributions.

    Args:
        transform (function): The transform.
        transform_derive (function): Derivative of the transform.
        untransform (function): Inverse of the transform.
        untransform_logdet (function): Log-determinant of the Jacobian of the inverse
            transform.

    Attributes:
        transform (function): The transform.
        transform_derive (function): Derivative of the transform.
        untransform (function): Inverse of the transform.
        untransform_logdet (function): Log-determinant of the Jacobian of the inverse
            transform.
    """

    def __init__(
        self,
        transform,
        transform_deriv,
        untransform,
        untransform_logdet,
    ):
        self.transform = transform
        self.transform_deriv = transform_deriv
        self.untransform = untransform
        self.untransform_logdet = untransform_logdet

    def __call__(self, dist):
        return TransformedMultiOutputDistribution(dist, self)

    @classmethod
    def exp(cls):
        """Construct the `exp` transform."""

        def transform(x):
            return B.exp(x)

        def transform_deriv(x):
            return B.exp(x)

        def untransform(y):
            return B.log(y)

        def untransform_logdet(y):
            return -B.log(y)

        return cls(
            transform=transform,
            transform_deriv=transform_deriv,
            untransform=untransform,
            untransform_logdet=untransform_logdet,
        )

    @classmethod
    def softplus(cls):
        """Construct the `softplus` transform."""

        def transform(x):
            return B.log(1 + B.exp(x))

        def transform_deriv(x):
            return B.exp(x) / (1 + B.exp(x))

        def untransform(y):
            return B.log(B.exp(y) - 1)

        def untransform_logdet(y):
            return B.exp(y) / (B.exp(y) - 1)

        return cls(
            transform=transform,
            transform_deriv=transform_deriv,
            untransform=untransform,
            untransform_logdet=untransform_logdet,
        )

    @classmethod
    def bounded(cls, lower, upper):
        """Construct a transform for a bounded variable.

        Args:
            lower (scalar): Lower bound.
            upper (scalar): Upper bound.
        """

        def transform(x):
            return lower + (upper - lower) / (1 + B.exp(-x))

        def transform_deriv(x):
            denom = 1 + B.exp(-x)
            return (upper - lower) * B.exp(-x) / (denom * denom)

        def untransform(y):
            return B.log(y - lower) - B.log(upper - y)

        def untransform_logdet(y):
            return B.log(1 / (y - lower) + 1 / (upper - y))

        return cls(
            transform=transform,
            transform_deriv=transform_deriv,
            untransform=untransform,
            untransform_logdet=untransform_logdet,
        )


class TransformedMultiOutputDistribution(AbstractMultiOutputDistribution):
    """A transformed multi-output distribution.

    Args:
        dist (:class:`.AbstractMultiOutputDistribution`): Transformed distribution.
        transform (:class:`.Transform`): Transform.

    Attributes:
        dist (:class:`.AbstractMultiOutputDistribution`): Transformed distribution.
        transform (:class:`.Transform`): Transform.
    """

    def __init__(self, dist, transform):
        self.dist = dist
        self.transform = transform

    def __repr__(self):
        return (
            f"<TransformedMultiOutputDistribution:\n"
            + indented_kv("dist", repr(self.dist), suffix="\n")
            + indented_kv("transform", repr(self.transform), suffix=">")
        )

    def __str__(self):
        return (
            f"<TransformedMultiOutputDistribution:\n"
            + indented_kv("dist", str(self.dist), suffix="\n")
            + indented_kv("transform", str(self.transform), suffix=">")
        )

    @property
    def noiseless(self):
        """:class:`.TransformedMultiOutputNormal`: Noiseless version of the
        distribution."""
        return TransformedMultiOutputDistribution(self.dist.noiseless, self.transform)

    @property
    def mean(self):
        return _map_aggregate(self.transform.transform, self.dist.mean)

    @property
    def var(self):
        def _var(m, v):
            deriv = self.transform.transform_deriv(m)
            return deriv * deriv * v

        return _map_aggregate(_var, self.dist.mean, self.dist.var)

    def logpdf(self, x):
        def _logdet_sum(x):
            return B.sum(
                self.transform.untransform_logdet(x),
                axis=tuple(range(-1 - data_dims(x), 0)),
            )

        logpdf = self.dist.logpdf(_map_aggregate(self.transform.untransform, x))
        logdet = _sum_aggregate(_map_aggregate(_logdet_sum, x))
        return logpdf + logdet

    def sample(self, *args, **kw_args):
        return _map_sample_output(
            partial(_map_aggregate, self.transform.transform),
            self.dist.sample(*args, **kw_args),
        )


@B.dispatch
def dtype(dist: TransformedMultiOutputDistribution):
    return B.dtype(dist.dist)


@B.dispatch
def cast(dtype: B.DType, dist: TransformedMultiOutputDistribution):
    return TransformedMultiOutputDistribution(B.cast(dtype, dist.dist), dist.transform)


@B.dispatch
def shape_batch(dist: TransformedMultiOutputDistribution):
    return B.shape_batch(dist.dist)


@_dispatch
def _map_aggregate(f, *xs):
    return f(*xs)


@_dispatch
def _map_aggregate(f, *xs: Aggregate):
    return Aggregate(*(f(*xis) for xis in zip(*xs)))


@_dispatch
def _sum_aggregate(x):
    return x


@_dispatch
def _sum_aggregate(x: Aggregate):
    return sum(x)
