import abc

__all__ = ["AbstractMultiOutputDistribution", "AbstractDistribution"]


class AbstractMultiOutputDistribution(metaclass=abc.ABCMeta):
    """An interface for multi-output distributions."""

    @abc.abstractmethod
    def __repr__(self):
        pass

    @abc.abstractmethod
    def __str__(self):
        pass

    @abc.abstractmethod
    def mean(self):
        """tensor: Marginal means of shape `(*b, c, n)`."""

    @abc.abstractmethod
    def var(self):
        """tensor: Marginal variances of shape `(*b, c, n)`."""

    @abc.abstractmethod
    def logpdf(self, x):
        """Compute the log-pdf at inputs `x`.

        Args:
            x (tensor): Inputs of shape `(*b, c, n)`.

        Returns:
            tensor: Log-pdfs of shape `(*b,)`.
        """

    @abc.abstractmethod
    def sample(self, state, num=1):
        """Sample from the distribution.

        Args:
            state (random state, optional): Random state.
            num (int, optional): Number of samples. Defaults to one.

        Returns:
            tensor: Samples of shape `(num, *b, c, n)` if `num > 1` and of shape .
                `(*b, c, n)` otherwise.
        """

    def kl(self, other):
        """Compute the KL-divergence with respect to another distribution.

        Args:
            other (:class:`.AbstractMultiOutputDistribution`): Other.

        Returns:
            tensor: KL-divergences `kl(self, other)` of shape `(*b,)`.
        """
        raise NotImplementedError(
            f"Cannot compute the KL-divergence between {self} and {other}."
        )

    def entropy(self):
        """Compute entropy.

        Returns:
            tensor: Entropies of shape `(*b,)`.
        """
        raise NotImplementedError(f"Cannot compute the entropy of {self}.")


class AbstractDistribution(metaclass=abc.ABCMeta):
    """An interface for distributions, mostly used for sampling in the data
    generators."""

    @abc.abstractmethod
    def sample(self, state, dtype, *shape):
        """Sample.

        Args:
            state (random state): Random state.
            dtype (dtype): Data type.
            *shape (int): Batch shape of the sample.

        Returns:
            random state: Random state.
            tensor: Sample of dtype `dtype` and shape `(*shape, *d)` where `d`
                specifies the dimensionality of the sample.
        """
