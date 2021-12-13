import lab as B
from matrix.util import indent

from . import _dispatch
from .util import abstract

__all__ = ["AbstractParallel"]


@abstract
class AbstractParallel:
    """A parallel of elements.

    Args:
        *elements (tuple): Objects to put in parallel.
    """

    def __init__(self, *elements):
        self.elements = elements

    def __call__(self, x):
        return AbstractParallel(*(e(x) for e in self.elements))

    def __iter__(self):
        return iter(self.elements)

    def __getitem__(self, item):
        return self.elements[item]

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            "Parallel(\n"
            + "".join([indent(repr(e).strip(), " " * 4) + ",\n" for e in self])
            + ")"
        )


@_dispatch
def code(
    p: AbstractParallel,
    xz: B.Numeric,
    z: B.Numeric,
    x: B.Numeric,
    **kw_args,
):
    xz, z = zip(*[code(pi, xz, z, x, **kw_args) for pi in p])
    return AbstractParallel(*xz), AbstractParallel(*z)


@_dispatch
def code(
    p: AbstractParallel,
    xz: B.Numeric,
    z: AbstractParallel,
    x: B.Numeric,
    **kw_args,
):
    xz, z = zip(*[code(pi, xz, zi, x, **kw_args) for (pi, zi) in zip(p, z)])
    return AbstractParallel(*xz), AbstractParallel(*z)


@_dispatch
def code(
    p: AbstractParallel,
    xz: AbstractParallel,
    z: AbstractParallel,
    x: B.Numeric,
    **kw_args,
):
    xz, z = zip(*[code(pi, xzi, zi, x, **kw_args) for (pi, xzi, zi) in zip(p, xz, z)])
    return AbstractParallel(*xz), AbstractParallel(*z)
