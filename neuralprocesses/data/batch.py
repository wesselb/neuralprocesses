import lab as B

from .. import _dispatch
from ..aggregate import Aggregate, AggregateInput
from ..mask import Masked
from ..augment import AugmentedInput

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
def batch_index(_: None, index):
    return None


@_dispatch
def batch_index(xt: AggregateInput, index):
    return AggregateInput(*((batch_index(xti, index), i) for xti, i in xt))


@_dispatch
def batch_index(yt: Aggregate, index):
    return Aggregate(*(batch_index(yti, index) for yti in yt))


@_dispatch
def batch_index(y: Masked, index):
    return Masked(batch_index(y.y, index), batch_index(y.mask, index))


@_dispatch
def batch_index(x: AugmentedInput, index):
    return AugmentedInput(
        batch_index(x.x, index),
        AugmentedInput(x.augmentation, index),
    )


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
    return _batch_yc(batch["contexts"][i][1])


@_dispatch
def _batch_yc(yc: B.Numeric):
    return yc[..., 0, :]


@_dispatch
def _batch_yc(yc: Masked):
    with B.on_device(yc.y):
        nan = B.to_active_device(B.cast(B.dtype(yc.y), B.nan))
    return B.where(yc.mask[..., 0, :] == 1, yc.y[..., 0, :], nan)


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
def _batch_xt(x: AggregateInput, i: int):
    return x[[xi[1] for xi in x].index(i)][0]


@_dispatch
def _batch_xt(x: AugmentedInput, i: int):
    return _batch_xt(x.x, i)


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
def _batch_yt(x: AggregateInput, y: Aggregate, i: int):
    return y[[xi[1] for xi in x].index(i)][..., 0, :]
