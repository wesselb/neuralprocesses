import lab as B
from plum import Tuple

from . import _dispatch

__all__ = ["Aggregate", "AggregateTargets"]


class Aggregate:
    """An ordered aggregate of things.

    Args:
        *elements (object): Elements in the aggregate.

    Attributes:
        elements (tuple): Elements in the aggregate.
    """

    def __init__(self, *elements):
        self.elements = elements

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, item):
        return self.elements[item]


@B.dispatch
def on_device(agg: Aggregate):
    return B.on_device(agg[0])


@B.dispatch
def dtype(agg: Aggregate):
    return B.dtype(*agg)


@B.dispatch
def cast(dtype: B.DType, agg: Aggregate):
    return Aggregate(*(B.cast(dtype, x) for x in agg))


class AggregateTargets(Aggregate):
    """An ordered aggregate of target inputs for specific outputs. This allow the user
    to specify different inputs for different outputs.

    Args:
        *elements (tuple[object, int]): A tuple of inputs and integers where the integer
            selects the particular output.
    """

    @_dispatch
    def __init__(self, *elements: Tuple[object, int]):
        super().__init__(*elements)


@B.dispatch
def on_device(agg: AggregateTargets):
    return B.on_device(agg[0][0])


@B.dispatch
def dtype(agg: AggregateTargets):
    return B.dtype(*(x for x, i in agg))


@B.dispatch
def cast(dtype: B.DType, agg: AggregateTargets):
    return Aggregate(*((B.cast(dtype, x), i) for x, i in agg))
