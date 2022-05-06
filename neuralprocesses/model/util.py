import lab as B
from matrix import Diagonal
from plum import Union
from stheno import Normal

from .. import _dispatch
from ..aggregate import Aggregate, AggregateInput
from ..dist import (
    AbstractMultiOutputDistribution,
    MultiOutputNormal,
    TransformedMultiOutputDistribution,
)
from ..parallel import Parallel

__all__ = ["sample", "fix_noise", "compress_contexts", "tile_for_sampling"]


@_dispatch
def sample(
    state: B.RandomState,
    x: AbstractMultiOutputDistribution,
    num: Union[B.Int, None] = None,
):
    """Sample an encoding:

    Args:
        state (random state): Random state.
        x (object): Encoding.
        num (int, optional): Number of samples.

    Returns:
        random state: Random state.
        object: Sample.
    """
    return x.sample(state, num=num)


@_dispatch
def sample(state: B.RandomState, x: Parallel, num: Union[B.Int, None] = None):
    samples = []
    for xi in x:
        state, s = sample(state, xi, num=num)
        samples.append(s)
    return state, Parallel(*samples)


@_dispatch
def fix_noise(d, epoch: None):
    """Fix the noise of a prediction in the first three epochs to `1e-4`.

    Args:
        d (:class:`neuralprocesses.dist.dist.AbstractMultiOutputDistribution`):
            Prediction.
        epoch (int or None): Epoch.

    Returns:
        :class:`neuralprocesses.dist.dist.AbstractMultiOutputDistribution`: Prediction
            with noise fixed.
    """
    return d


@_dispatch
def fix_noise(d: MultiOutputNormal, epoch: int):
    # Fix noise to `1e-4` in the first three epochs.
    if epoch < 3:
        with B.on_device(d.vectorised_normal.var_diag):
            d = MultiOutputNormal(
                d._mean,
                B.zeros(d._var),
                1e-4 * Diagonal(B.ones(d.vectorised_normal.var_diag)),
                d.shape,
            )
    return d


@_dispatch
def fix_noise(d: TransformedMultiOutputDistribution, epoch: int):
    return TransformedMultiOutputDistribution(
        fix_noise(d.dist, epoch),
        d.transform,
    )


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
