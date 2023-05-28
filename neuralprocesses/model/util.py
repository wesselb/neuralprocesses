import lab as B
from matrix import Diagonal

from .. import _dispatch
from ..aggregate import Aggregate, AggregateInput
from ..dist import (
    AbstractDistribution,
    MultiOutputNormal,
    SpikesSlab,
    TransformedMultiOutputDistribution,
)
from ..parallel import Parallel

__all__ = ["sample", "fix_noise", "compress_contexts", "tile_for_sampling"]


@_dispatch
def sample(
    state: B.RandomState,
    x: AbstractDistribution,
    *shape: B.Int,
):
    """Sample an encoding.

    Args:
        state (random state): Random state.
        x (object): Encoding.
        *shape (int): Batch shape of the sample.

    Returns:
        random state: Random state.
        object: Sample.
    """
    return x.sample(state, *shape)


@_dispatch
def sample(state: B.RandomState, x: Parallel, *shape: B.Int):
    samples = []
    for xi in x:
        state, s = sample(state, xi, *shape)
        samples.append(s)
    return state, Parallel(*samples)


@_dispatch
def fix_noise(d, value: None):
    """Fix the noise of a prediction.

    Args:
        d (:class:`neuralprocesses.dist.dist.AbstractDistribution`):
            Prediction.
        value (float or None): Value to fix it to.

    Returns:
        :class:`neuralprocesses.dist.dist.AbstractDistribution`: Prediction
            with noise fixed.
    """
    return d


@_dispatch
def fix_noise(d: MultiOutputNormal, value: float):
    with B.on_device(d.vectorised_normal.var_diag):
        return MultiOutputNormal(
            d._mean,
            B.zeros(d._var),
            value * Diagonal(B.ones(d.vectorised_normal.var_diag)),
            d.shape,
        )


@_dispatch
def fix_noise(d: TransformedMultiOutputDistribution, value: float):
    return TransformedMultiOutputDistribution(
        fix_noise(d.dist, value),
        d.transform,
    )


@_dispatch
def fix_noise(d: SpikesSlab, value: float):
    return d


@_dispatch
def compress_contexts(contexts: list):
    """Compress multiple context sets into a single `(x, y)` pair.

    Args:
        contexts (list): Context sets.

    Returns:
        input: Context inputs.
        object: Context outputs.
    """
    # Don't unnecessarily wrap things in a `Parallel`.
    if len(contexts) == 1:
        return contexts[0]
    else:
        return (
            Parallel(*(c[0] for c in contexts)),
            Parallel(*(c[1] for c in contexts)),
        )


@_dispatch
def tile_for_sampling(x: B.Numeric, num_samples: int):
    """Tile to setup batching to produce multiple samples.

    Args:
        x (object): Object to tile.
        num_samples (int): Number of samples.

    Returns:
        object: `x` tiled `num_samples` number of times.
    """
    return B.tile(x[None, ...], num_samples, *((1,) * B.rank(x)))


@_dispatch
def tile_for_sampling(y: Aggregate, num_samples: int):
    return Aggregate(*(tile_for_sampling(yi, num_samples) for yi in y))


@_dispatch
def tile_for_sampling(x: AggregateInput, num_samples: int):
    return Aggregate(*((tile_for_sampling(xi, num_samples), i) for xi, i in x))


@_dispatch
def tile_for_sampling(contexts: list, num_samples: int):
    return [
        (tile_for_sampling(xi, num_samples), tile_for_sampling(yi, num_samples))
        for xi, yi in contexts
    ]
