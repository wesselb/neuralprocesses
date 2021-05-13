import lab as B

from . import _dispatch

__all__ = ["Parallel"]


class Parallel:
    """A parallel of elements.

    Args:
        *elements (tuple): Objects to put in parallel.
    """

    def __init__(self, *elements):
        self.elements = elements

    def __call__(self, x):
        return Parallel(*(e(x) for e in self.elements))


@_dispatch
def code(p: Parallel, xz: B.Numeric, z: B.Numeric, x: B.Numeric, **kw_args):
    xz, z = zip([code(pi, xz, z, x, **kw_args) for pi in p.elements])
    return Parallel(xz), Parallel(z)


@_dispatch
def code(p: Parallel, xz: B.Numeric, z: Parallel, x: B.Numeric, **kw_args):
    xz, z = zip(
        [code(pi, xz, zi, x, **kw_args) for (pi, zi) in zip(p.elements, z.elements)]
    )
    return Parallel(xz), Parallel(z)


@_dispatch
def code(p: Parallel, xz: Parallel, z: Parallel, x: B.Numeric, **kw_args):
    xz, z = zip(
        [
            code(pi, xzi, zi, x, **kw_args)
            for (pi, xzi, zi) in zip(p.elements, xz.elements, z.elements)
        ]
    )
    return Parallel(xz), Parallel(z)
