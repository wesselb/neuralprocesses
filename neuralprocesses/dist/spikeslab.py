import lab as B
from matrix.shape import broadcast
from wbml.util import indented_kv

from .dist import AbstractDistribution, shape_batch
from .. import _dispatch
from ..aggregate import Aggregate
from ..mask import Masked

__all__ = ["SpikesSlab"]


class SpikesSlab(AbstractDistribution):
    """Spikes-and-slab distribution.

    Args:
        spikes (vector): Spikes.
        slab (:class:`neuralprocesses.dist.dist.AbstractDistribution`): Slab.
        logprobs (tensor): Log-probabilities for the spikes and the slab with the
            log-probability for the slab last.
        epsilon (float, optional): Tolerance for equality checking. Defaults to `1e-6`.

    Attributes:
        spikes (vector): Spikes.
        slab (:class:`neuralprocesses.dist.dist.AbstractDistribution`): Slab.
        logprobs (tensor): Log-probabilities for the spikes and the slab with the
            log-probability for the slab last.
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
        # TODO: What to do here?
        return self

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
        logprob_cat, ind_slab = self.logpdf_cat(self.logprobs, self.d, x)
        logpdf_slab = self.slab.logpdf(_mask(x, ind_slab))
        return logprob_cat + logpdf_slab

    @_dispatch
    def logpdf_cat(self, logprobs: Aggregate, d: B.Int, x: Aggregate):
        li, ii = zip(
            *(self.logpdf_cat(li, di, xi) for li, di, xi in zip(logprobs, d, x))
        )
        return Aggregate(*li), Aggregate(*ii)

    @_dispatch
    def logpdf_cat(self, logprobs: B.Numeric, d: B.Int, x: B.Numeric):
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
        logprob_cat = B.sum(full_indicator * logprobs, axis=-1)
        # Sum over the sample dimensions.
        dims = tuple(range(B.rank(logprob_cat)))[-d:]
        return B.sum(logprob_cat, axis=dims), ind_slab

    @_dispatch
    def sample(self, state: B.RandomState, dtype: B.DType, *shape):
        state, slab_sample = self.slab.sample(state, dtype, *shape)
        return self.sample_spikes(
            self.logprobs,
            slab_sample,
            state,
            dtype,
            *shape,
        )

    @_dispatch
    def sample_spikes(
        self,
        logprobs: Aggregate,
        sample_slab: Aggregate,
        state: B.RandomState,
        dtype: B.DType,
        *shape,
    ):
        samples = []
        for li, si in zip(logprobs, sample_slab):
            state, sample = self.sample_spikes(li, si, state, dtype, *shape)
            samples.append(sample)
        return state, Aggregate(*samples)

    @_dispatch
    def sample_spikes(
        self,
        logprobs: B.Numeric,
        sample_slab: B.Numeric,
        state: B.RandomState,
        dtype: B.DType,
        *shape,
    ):
        with B.on_device(logprobs):
            # Sample the categorical variable.
            state, inds = B.randcat(state, B.exp(logprobs), *shape)

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
def shape_batch(dist: SpikesSlab):
    return broadcast(shape_batch(dist.slab), shape_batch(dist, dist.logprobs, dist.d))


@shape_batch.dispatch
def shape_batch(dist: SpikesSlab, logprobs: B.Numeric, d: B.Int):
    # `logprobs` has one extra dimension!
    return B.shape(logprobs)[: -(d + 1)]


@shape_batch.dispatch
def shape_batch(dist: SpikesSlab, logprobs: Aggregate, d: Aggregate):
    return broadcast(*(shape_batch(dist, li, di) for li, di in zip(logprobs, d)))


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
