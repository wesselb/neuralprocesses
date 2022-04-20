from matrix.util import indent

from . import _dispatch
from .util import register_module

__all__ = ["Parallel", "broadcast_coder_over_parallel"]


@register_module
class Parallel:
    """A parallel of elements.

    Args:
        *elements (object): Objects to put in parallel.

    Attributes:
        elements (tuple): Objects in parallel.
    """

    def __init__(self, *elements):
        try:
            self.elements = self.nn.ModuleList(elements)
        except Exception:
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
def code(p: Parallel, xz, z, x, **kw_args):
    xz, z = zip(*[code(pi, xz, z, x, **kw_args) for pi in p])
    return Parallel(*xz), Parallel(*z)


@_dispatch
def code(p: Parallel, xz, z: Parallel, x, **kw_args):
    xz, z = zip(*[code(pi, xz, zi, x, **kw_args) for (pi, zi) in zip(p, z)])
    return Parallel(*xz), Parallel(*z)


def broadcast_coder_over_parallel(Coder):
    @_dispatch
    def code(p: Coder, xz: Parallel, z: Parallel, x, **kw_args):
        xz, z = zip(*[code(p, xzi, zi, x, **kw_args) for (xzi, zi) in zip(xz, z)])
        return Parallel(*xz), Parallel(*z)


@_dispatch
def code(p: Parallel, xz: Parallel, z: Parallel, x, **kw_args):
    xz, z = zip(*[code(pi, xzi, zi, x, **kw_args) for (pi, xzi, zi) in zip(p, xz, z)])
    return Parallel(*xz), Parallel(*z)


@_dispatch
def data_dims(x: Parallel):
    dims = [data_dims(xi) for xi in x]
    if not all(d == dims[0] for d in dims[1:]):
        raise RuntimeError("Inconsistent data dimensions.")
    return dims[0]
