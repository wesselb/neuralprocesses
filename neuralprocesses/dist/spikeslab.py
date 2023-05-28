import lab as B
from matrix.shape import broadcast
from plum import parametric
from wbml.util import indented_kv

from .. import _dispatch
from ..aggregate import Aggregate
from ..mask import Masked
from .dirac import Dirac
from .dist import AbstractDistribution, shape_batch

__all__ = ["SpikesSlab"]


@parametric
class SpikesSlab(AbstractDistribution):
    """Spikes-and-slab distribution.

    Args:
        spikes (vector): Spikes.
        slab (:class:`neuralprocesses.dist.dist.AbstractDistribution`): Slab.
        logprobs (tensor): Log-probabilities for the spikes and the slab with the
            log-probability for the slab last.
        d (int): Dimensionality of the data.
        epsilon (float, optional): Tolerance for equality checking. Defaults to `1e-6`.

    Attributes:
        spikes (vector): Spikes.
        slab (:class:`neuralprocesses.dist.dist.AbstractDistribution`): Slab.
        logprobs (tensor): Log-probabilities for the spikes and the slab with the
            log-probability for the slab last.
        d (int): Dimensionality of the data.
        epsilon (float): Tolerance for equality checking.
    """

    @_dispatch
    def __init__(
        self,
        spikes,
        slab: AbstractDistribution,
        logprobs,
        d,
        *,
        epsilon: float = 1e-6,
    ):
        self.spikes = spikes
        self.slab = slab
        # Normalise the probabilities to ensure that the log-pdf computation is correct.
        self.logprobs = B.subtract(
            logprobs,
            B.logsumexp(logprobs, axis=-1, squeeze=False),
        )
        self.d = d
        self.epsilon = epsilon

    def __repr__(self):
        return (  # Comment to preserve formatting.
            f"<SpikesSlab:"
            f" spikes={self.spikes!r},"
            f" epsilon={self.epsilon!r},"
            + indented_kv("slab", repr(self.slab), suffix="\n")
            + indented_kv("logprobs", repr(self.logprobs), suffix=">")
        )

    def __str__(self):
        return (  # Comment to preserve formatting.
            f"<SpikeSlab:"
            f" spikes={self.spikes},"
            f" epsilon={self.epsilon},"
            + indented_kv("slab", str(self.slab), suffix="\n")
            + indented_kv("logprobs", str(self.logprobs), suffix=">")
        )

    @property
    def noiseless(self):
        return Dirac(self.mean, self.d)

    @property
    def m1(self):
        m1_spikes = self.spikes
        m1_slab = self.slab.m1
        m1 = _spikeslab_concat(m1_spikes, m1_slab)
        return B.sum(B.multiply(B.exp(self.logprobs), m1), axis=-1)

    @property
    def m2(self):
        m2_spikes = self.spikes * self.spikes
        m2_slab = self.slab.m2
        m2 = _spikeslab_concat(m2_spikes, m2_slab)
        return B.sum(B.multiply(B.exp(self.logprobs), m2), axis=-1)

    @_dispatch
    def logpdf(self, x):
        logprob_cat, ind_slab = self.logpdf_cat(x)
        logpdf_slab = self.slab.logpdf(_mask(x, ind_slab))
        return logprob_cat + logpdf_slab

    @_dispatch
    def logpdf_cat(
        self: "SpikesSlab[B.Numeric, AbstractDistribution, Aggregate, Aggregate]",
        x: Aggregate,
    ):
        li, ii = zip(
            *(
                SpikesSlab(self.spikes, self.slab, li, di).logpdf_cat(xi)
                for li, di, xi in zip(self.logprobs, self.d, x)
            )
        )
        return sum(li, 0), Aggregate(*ii)

    @_dispatch
    def logpdf_cat(
        self: "SpikesSlab[B.Numeric, AbstractDistribution, B.Numeric, B.Int]",
        x: B.Numeric,
    ):
        # Construct indicators which filter out the spikes.
        ind_spikes = B.cast(
            B.dtype(x),
            B.abs(
                B.expand_dims(self.spikes, axis=0, times=B.rank(x))
                - B.expand_dims(x, axis=-1, times=1)
            )
            < self.epsilon,
        )
        # Construct an indicator for the slab.
        with B.on_device(x):
            any_spike = B.minimum(B.sum(ind_spikes, axis=-1), B.one(x))
            ind_slab = B.one(x) - any_spike
        # Compute the log-probability of the categorical random variable.
        full_indicator = B.concat(ind_spikes, ind_slab[..., None], axis=-1)
        logprob_cat = B.sum(full_indicator * self.logprobs, axis=-1)
        # Sum over the sample dimensions.
        dims = tuple(range(B.rank(logprob_cat)))[-self.d :]
        return B.sum(logprob_cat, axis=dims), ind_slab

    @_dispatch
    def sample(self, state: B.RandomState, dtype: B.DType, *shape):
        state, sample_slab = self.slab.sample(state, dtype, *shape)
        return self.sample_spikes(
            sample_slab,
            state,
            dtype,
            *shape,
        )

    @_dispatch
    def sample_spikes(
        self: "SpikesSlab[B.Numeric, AbstractDistribution, Aggregate, Aggregate]",
        sample_slab: Aggregate,
        state: B.RandomState,
        dtype: B.DType,
        *shape,
    ):
        samples = []
        for li, di, si in zip(self.logprobs, self.d, sample_slab):
            dist = SpikesSlab(self.spikes, self.slab, li, di, epsilon=self.epsilon)
            state, sample = dist.sample_spikes(si, state, dtype, *shape)
            samples.append(sample)
        return state, Aggregate(*samples)

    @_dispatch
    def sample_spikes(
        self: "SpikesSlab[B.Numeric, AbstractDistribution, B.Numeric, B.Int]",
        sample_slab: B.Numeric,
        state: B.RandomState,
        dtype: B.DType,
        *shape,
    ):
        with B.on_device(self.logprobs):
            # Sample the categorical variable.
            state, inds = B.randcat(state, B.exp(self.logprobs), *shape)

            # Construct indicators to filter the spikes or slab samples.
            spikes_and_zero = B.concat(self.spikes, B.zero(self.spikes)[None])
            zeros_and_one = B.concat(B.zeros(self.spikes), B.one(self.spikes)[None])
            sample_spikes = B.reshape(
                B.take(spikes_and_zero, B.flatten(inds)),
                *B.shape(inds),
            )
            slab_indicator = B.reshape(
                B.take(zeros_and_one, B.flatten(inds)),
                *B.shape(inds),
            )
        # Assemble sample.
        return state, sample_spikes + slab_indicator * sample_slab


@B.dtype.dispatch
def dtype(d: SpikesSlab):
    return B.dtype(d.spikes, d.spikes, d.logprobs)


@shape_batch.dispatch
def shape_batch(dist: SpikesSlab[B.Numeric, AbstractDistribution, B.Numeric, B.Int]):
    shape_slab = shape_batch(dist.slab)
    # `logprobs` has one extra dimension!
    shape_spikes = B.shape(dist.logprobs)[: -(dist.d + 1)]
    return broadcast(shape_slab, shape_spikes)


@shape_batch.dispatch
def shape_batch(
    dist: SpikesSlab[B.Numeric, AbstractDistribution, Aggregate, Aggregate]
):
    return broadcast(
        *(
            shape_batch(SpikesSlab(dist.spikes, dist.slab, li, di))
            for li, di in zip(dist.logprobs, dist.d)
        )
    )


@_dispatch
def _spikeslab_concat(spikes, slab):
    return B.concat(
        B.repeat(spikes, *B.shape(slab)),
        B.expand_dims(slab, axis=-1),
        axis=-1,
    )


@_dispatch
def _spikeslab_concat(spikes, slab: Aggregate):
    return Aggregate(*(_spikeslab_concat(spikes, si) for si in slab))


@_dispatch
def _mask(x, mask):
    return Masked(x, mask)


@_dispatch
def _mask(x: Aggregate, mask: Aggregate):
    return Aggregate(*(_mask(xi, mi) for xi, mi in zip(x, mask)))
