import abc

import lab as B

from .. import _dispatch

__all__ = ["AbstractDistribution", "AbstractMultiOutputDistribution", "shape_batch"]


class AbstractDistribution(metaclass=abc.ABCMeta):
    """An interface for distributions."""

    @abc.abstractmethod
    def __repr__(self):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    @_dispatch
    def sample(self, state: B.RandomState, dtype: B.DType, *shape):
        """Sample from the distribution.

        Args:
            state (random state, optional): Random state.
            dtype (dtype, optional): Data type.
            *shape (int): Batch shape of the sample.

        Returns:
            state (random state, optional): Random state.
            tensor: Samples of shape `(*shape, *d)` where typically `d = (*b, c, n)`.
        """
        raise NotImplementedError(f"{self} cannot be sampled.")

    @_dispatch
    def sample(self, dtype: B.DType, *shape):
        state = B.global_random_state(dtype)
        state, sample = self.sample(state, dtype, *shape)
        B.set_global_random_state(state)
        return sample

    @_dispatch
    def sample(self, state: B.RandomState, *shape):
        return self.sample(state, B.dtype(self), *shape)

    @_dispatch
    def sample(self, *shape):
        return self.sample(B.dtype(self), *shape)

    def logpdf(self, x):
        """Compute the log-pdf at inputs `x`.

        Args:
            x (tensor): Inputs of shape `(*b, *d)` where `b` is the batch shape and.
                `d` the dimensionality of the random variable.

        Returns:
            tensor: Log-pdfs of shape `(*b,)`.
        """
        raise NotImplementedError(f"Log-pdf of {self} cannot be computed.")

    @property
    def mean(self):
        """tensor: Mean."""
        return self.m1

    @property
    def var(self):
        """tensor: Marginal variance."""
        m1 = self.m1
        return B.subtract(self.m2, B.multiply(m1, m1))

    @property
    def m1(self):
        """tensor: First moment."""
        return self.mean

    @property
    def m2(self):
        """tensor: Second moment."""
        mean = self.mean
        return B.add(self.var, B.multiply(mean, mean))

    def kl(self, other):
        """Compute the KL-divergence with respect to another distribution.

        Args:
            other (:class:`.AbstractDistribution`): Other.

        Returns:
            tensor: KL-divergences `kl(self, other)`.
        """
        raise NotImplementedError(
            f"Cannot compute the KL-divergence between {self} and {other}."
        )

    def entropy(self):
        """Compute entropy.

        Returns:
            tensor: Entropy.
        """
        raise NotImplementedError(f"Cannot compute the entropy of {self}.")


# Support this for backwards compatibility.
AbstractMultiOutputDistribution = AbstractDistribution


@_dispatch
def shape_batch(dist: AbstractDistribution):
    """Determine the batch shape of a distribution.

    Args:
        dist (:class:`.AbstractDistribution`): Distribution.
        *dims (int, optional): Dimensions to get.

    Returns:
        shape or int: Batch shape of the prediction of `dist`.
    """
    raise NotImplementedError(f"Cannot determine the batch shape of {dist}.")


@_dispatch
def shape_batch(dist, *dims: B.Int):
    dist_shape_batch = shape_batch(dist)
    return B.squeeze(tuple(dist_shape_batch[d] for d in dims))
