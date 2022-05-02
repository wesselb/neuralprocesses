import lab as B
from functools import partial
from wbml.util import indented_kv

from .. import _dispatch

from .dist import AbstractMultiOutputDistribution
from .normal import _map_sample_output
from ..util import register_module
from ..aggregate import Aggregate
from ..datadims import data_dims

__all__ = ["Transform"]


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
    def positive(cls):
        """Construct the `exp` transform."""

        def transform(x):
            return B.exp(x)

        def transform_deriv(x):
            return B.exp(x)

        def untransform(x):
            return B.log(x)

        def untransform_logdet(x):
            return -B.log(x)

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

        def untransform(x):
            return B.log(x - lower) - B.log(upper - x)

        def untransform_logdet(x):
            return B.log(1 / (x - lower) + 1 / (upper - x))

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
