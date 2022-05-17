import lab as B

from . import _dispatch
from .aggregate import Aggregate, AggregateInput
from .parallel import Parallel
from .augment import AugmentedInput

__all__ = ["data_dims"]


@_dispatch
def data_dims(x: B.Numeric):
    """Check how many data dimensions the encoding corresponding to an input has.

    Args:
        x (input): Input.

    Returns:
        int: Number of data dimensions.
    """
    return 1


@_dispatch
def data_dims(x: None):
    return 1


@_dispatch
def data_dims(x: tuple):
    return len(x)


@_dispatch
def data_dims(x: Parallel):
    return _data_dims_merge(*(data_dims(xi) for xi in x))


@_dispatch
def _data_dims_merge(d1, d2, d3, *ds):
    d = _data_dims_merge(d1, d2)
    for di in (d3,) + ds:
        d = _data_dims_merge(d, di)
    return d


@_dispatch
def _data_dims_merge(d):
    return d


@_dispatch
def _data_dims_merge(d1, d2):
    if d1 == d2:
        return d1
    else:
        raise RuntimeError(f"Cannot reconcile data dimensionalities {d1} and {d2}.")


@_dispatch
def data_dims(x: AggregateInput):
    return Aggregate(*(data_dims(xi) for xi, _ in x))


@_dispatch
def _data_dims_merge(d1: Aggregate, d2: Aggregate):
    return Aggregate(*(_data_dims_merge(d1i, d2i) for d1i, d2i in zip(d1, d2)))


@_dispatch
def _data_dims_merge(d1: Aggregate, d2):
    return _data_dims_merge(*(_data_dims_merge(d1i, d2) for d1i in d1))


@_dispatch
def _data_dims_merge(d1, d2: Aggregate):
    return _data_dims_merge(*(_data_dims_merge(d1, d2i) for d2i in d2))


@_dispatch
def data_dims(x: AugmentedInput):
    return data_dims(x.x)
