import lab as B
from matrix.util import indent

from . import _dispatch
from .util import register_module

__all__ = ["Parallel"]


@register_module
class Parallel:
    """A parallel of elements.

    Args:
        *elements (tuple): Objects to put in parallel.
    """

    def __init__(self, *elements):
        try:
            self.elements = self.nn.ModuleList(elements)
        except AttributeError:
            self.elements = elements

    def __call__(self, x):
        return Parallel(*(e(x) for e in self.elements))

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
def code(p: Parallel, xz: B.Numeric, z: B.Numeric, x: B.Numeric, **kw_args):
    xz, z = zip(*[code(pi, xz, z, x, **kw_args) for pi in p])
    return Parallel(*xz), Parallel(*z)


@_dispatch
def code(p: Parallel, xz: B.Numeric, z: Parallel, x: B.Numeric, **kw_args):
    xz, z = zip(*[code(pi, xz, zi, x, **kw_args) for (pi, zi) in zip(p, z)])
    return Parallel(*xz), Parallel(*z)


@_dispatch
def code(p: Parallel, xz: Parallel, z: Parallel, x: B.Numeric, **kw_args):
    xz, z = zip(*[code(pi, xzi, zi, x, **kw_args) for (pi, xzi, zi) in zip(p, xz, z)])
    return Parallel(*xz), Parallel(*z)
