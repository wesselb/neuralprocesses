import lab as B
from matrix import Diagonal
from stheno import Normal

from .. import _dispatch
from ..dist import (
    AbstractMultiOutputDistribution,
    MultiOutputNormal,
    TransformedMultiOutputDistribution,
)
from ..parallel import Parallel

__all__ = ["sample", "fix_noise"]


@_dispatch
def sample(state: B.RandomState, x: AbstractMultiOutputDistribution, num: B.Int = 1):
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
def sample(state: B.RandomState, x: Parallel, num: B.Int = 1):
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
        var_diag = d.normal.var_diag
        with B.on_device(var_diag):
            var = Diagonal(1e-4 * B.ones(var_diag))
        d = MultiOutputNormal(Normal(d.normal.mean, var), d.shape)
    return d


@_dispatch
def fix_noise(d: TransformedMultiOutputDistribution, epoch: int):
    return TransformedMultiOutputDistribution(
        fix_noise(d.dist, epoch),
        d.transform,
    )
