from typing import Tuple

import lab as B

from . import _dispatch

__all__ = ["Aggregate", "AggregateInput"]


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


def _assert_equal_lengths(*elements):
    if any(len(elements[0]) != len(e) for e in elements[1:]):
        raise ValueError("Aggregates have unequal lengths.")


def _map_f(name, num_args):
    method = getattr(B, name)

    if num_args == 1:

        @method.dispatch
        def f(a: Aggregate, **kw_args):
            return Aggregate(*(getattr(B, name)(ai, **kw_args) for ai in a))

    elif num_args == 2:

        @method.dispatch
        def f(a: Aggregate, b: Aggregate, **kw_args):
            _assert_equal_lengths(a, b)
            return Aggregate(
                *(getattr(B, name)(ai, bi, **kw_args) for ai, bi in zip(a, b))
            )

    elif num_args == "*":

        @method.dispatch
        def f(*args: Aggregate, **kw_args):
            _assert_equal_lengths(*args)
            return Aggregate(*(getattr(B, name)(*xs, **kw_args) for xs in zip(*args)))

    else:
        raise ValueError(f"Invalid number of arguments {num_args}.")


_map_f("expand_dims", 1)
_map_f("exp", 1)
_map_f("one", 1)
_map_f("zero", 1)
_map_f("mean", 1)
_map_f("sum", 1)
_map_f("logsumexp", 1)

_map_f("add", 2)
_map_f("subtract", 2)
_map_f("multiply", 2)
_map_f("divide", 2)

_map_f("stack", "*")
_map_f("concat", "*")
_map_f("squeeze", "*")


@B.dispatch
def max(a: Aggregate):
    return B.max(B.stack(*(B.max(ai) for ai in a)))


@B.dispatch
def min(a: Aggregate):
    return B.min(B.stack(*(B.min(ai) for ai in a)))


class AggregateInput:
    """An ordered aggregate of inputs for specific outputs. This allow the user to
    specify different inputs for different outputs.

    Args:
        *elements (tuple[object, int]): A tuple of inputs and integers where the integer
            selects the particular output.

    Attributes:
        elements (tuple[object, int]): Elements in the aggregate input.
    """

    @_dispatch
    def __init__(self, *elements: Tuple[object, int]):
        self.elements = elements

    def __iter__(self):
        return iter(self.elements)

    def __len__(self):
        return len(self.elements)

    def __getitem__(self, item):
        return self.elements[item]


@B.dispatch
def on_device(agg: AggregateInput):
    return B.on_device(agg[0][0])


@B.dispatch
def dtype(agg: AggregateInput):
    return B.dtype(*(x for x, i in agg))


@B.dispatch
def cast(dtype: B.DType, agg: AggregateInput):
    return Aggregate(*((B.cast(dtype, x), i) for x, i in agg))
