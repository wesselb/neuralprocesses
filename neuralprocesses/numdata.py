import lab as B
from . import _dispatch
import numpy as np

from .aggregate import Aggregate, AggregateTargets
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
    # Also count the channels dimension!
    return np.prod(B.shape(y)[-d - 1 :])


@_dispatch
def num_data(x: AggregateTargets, y: Aggregate):
    return sum([num_data(xi, yi) for (xi, i), yi in zip(x, y)])
