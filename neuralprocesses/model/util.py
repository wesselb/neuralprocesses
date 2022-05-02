import lab as B
from matrix import Diagonal
from plum import Union
from stheno import Normal

from .. import _dispatch
from ..dist import AbstractMultiOutputDistribution, MultiOutputNormal
from ..parallel import Parallel

__all__ = ["sample", "fix_noise"]


@_dispatch
def sample(state: B.RandomState, x: AbstractMultiOutputDistribution, num: B.Int = 1):
    return x.sample(state, num=num)


@_dispatch
def sample(state: B.RandomState, x: Parallel, num: B.Int = 1):
    samples = []
    for xi in x:
        state, s = sample(state, xi, num=num)
        samples.append(s)
    return state, Parallel(*samples)


@_dispatch
def fix_noise(d: AbstractMultiOutputDistribution, epoch: Union[int, None]):
    # Cannot handle the general case.
    return d


@_dispatch
def fix_noise(d: MultiOutputNormal, epoch: Union[int, None]):
    if epoch is not None and epoch < 3:
        # Fix noise to `1e-4`.
        var_diag = d.normal.var_diag
        with B.on_device(var_diag):
            var = Diagonal(1e-4 * B.ones(var_diag))
        d = MultiOutputNormal(Normal(d.normal.mean, var), d.shape)
    return d
