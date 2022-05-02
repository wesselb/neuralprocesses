import lab as B

from .. import _dispatch
from ..aggregate import Aggregate, AggregateTargets

__all__ = [
    "batch_index",
    "batch_xc",
    "batch_yc",
    "batch_xt",
    "batch_yt",
]


@_dispatch
def batch_index(batch: dict, index):
    """Index into the tensors of a batch.

    Args:
        batch (dict): Batch dictionary.
        index (object): Index.

    Returns:
        dict: `batch` indexed at `index`.
    """
    return {k: batch_index(v, index) for k, v in batch.items()}


@_dispatch
def batch_index(x: B.Numeric, index):
    return x[index]


@_dispatch
def batch_index(t: tuple, index):
    return tuple(batch_index(ti, index) for ti in t)


@_dispatch
def batch_index(t: list, index):
    return [batch_index(ti, index) for ti in t]


@_dispatch
def batch_index(xt: AggregateTargets, index):
    return AggregateTargets(*((batch_index(xti, index), i) for xti, i in xt))


@_dispatch
def batch_index(yt: Aggregate, index):
    return Aggregate(*(batch_index(yti, index) for yti in yt))


@_dispatch
def batch_xc(batch: dict, i: int):
    """Get the context inputs for a particular output dimension.

    Args:
        batch (dict): Batch dictionary.
        i (int): Index of output.

    Returns:
        tensor: Context inputs.
    """
    return batch["contexts"][i][0]


@_dispatch
def batch_yc(batch: dict, i: int):
    """Get the context outputs for a particular output dimension.

    Args:
        batch (dict): Batch dictionary.
        i (int): Index of output.

    Returns:
        tensor: Context outputs.
    """
    return batch["contexts"][i][1][..., 0, :]


@_dispatch
def batch_xt(batch: dict, i: int):
    """Get the target inputs for a particular output dimension.

    Args:
        batch (dict): Batch dictionary.
        i (int): Index of output.

    Returns:
        tensor: Target inputs.
    """
    return _batch_xt(batch["xt"], i)


@_dispatch
def _batch_xt(x: B.Numeric, i: int):
    return x


@_dispatch
def _batch_xt(x: AggregateTargets, i: int):
    return x[[xi[1] for xi in x].index(i)][0]


@_dispatch
def batch_yt(batch: dict, i: int):
    """Get the target outputs for a particular output dimension.

    Args:
        batch (dict): Batch dictionary.
        i (int): Index of output.

    Returns:
        tensor: Target outputs.
    """
    return _batch_yt(batch["xt"], batch["yt"], i)


@_dispatch
def _batch_yt(x: B.Numeric, y: B.Numeric, i: int):
    return y[..., i, :]


@_dispatch
def _batch_yt(x: AggregateTargets, y: Aggregate, i: int):
    return y[[xi[1] for xi in x].index(i)][..., 0, :]
