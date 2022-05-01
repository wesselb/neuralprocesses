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
def dtype(agg: Aggregate):
    return B.dtype(*agg)


@B.dispatch
def shape(agg: Aggregate):
    return Aggregate(*(B.shape(x) for x in agg))


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
