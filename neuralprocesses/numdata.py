import lab as B

from . import _dispatch
from .aggregate import Aggregate, AggregateInput
from .datadims import data_dims

__all__ = ["num_data"]


@_dispatch
def num_data(x, y: B.Numeric):
    """Count the number of data points.

    Args:
        x (input): Inputs.
        y (object): Outputs.

    Returns:
        int: Number of data points.
    """
    d = data_dims(x)
    available = B.cast(B.dtype_float(y), ~B.isnan(y))
    # Sum over the channel dimension and over all data dimensions.
    return B.sum(available, axis=tuple(range(-d - 1, 0)))


@_dispatch
def num_data(x: AggregateInput, y: Aggregate):
    return sum([num_data(xi, yi) for (xi, i), yi in zip(x, y)])
