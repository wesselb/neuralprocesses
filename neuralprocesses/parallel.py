import lab as B
from matrix.util import indent

from . import _dispatch
from .util import is_framework_module, register_module

__all__ = [
    "Parallel",
    "broadcast_coder_over_parallel",
]


@register_module
class Parallel:
    """A parallel of elements.

    Args:
        *elements (object): Objects to put in parallel.

    Attributes:
        elements (tuple): Objects in parallel.
    """

    def __init__(self, *elements):
        if any(is_framework_module(element) for element in elements):
            self.elements = self.nn.ModuleList(elements)
        else:
            self.elements = elements

    def __call__(self, x):
        return Parallel(*(e(x) for e in self.elements))

    def __getitem__(self, item):
        return self.elements[item]

    def __len__(self):
        return len(self.elements)

    def __iter__(self):
        return iter(self.elements)

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            "Parallel(\n"
            + "".join([indent(repr(e).strip(), " " * 4) + ",\n" for e in self])
            + ")"
        )


@B.cast.dispatch
def cast(dtype, x: Parallel):
    return Parallel(*(B.cast(dtype, xi) for xi in x))


@_dispatch
def code(p: Parallel, xz, z, x, **kw_args):
    xz, z = zip(*[code(pi, xz, z, x, **kw_args) for pi in p])
    return Parallel(*xz), Parallel(*z)


@_dispatch
def code(p: Parallel, xz, z: Parallel, x, **kw_args):
    xz, z = zip(*[code(pi, xz, zi, x, **kw_args) for (pi, zi) in zip(p, z)])
    return Parallel(*xz), Parallel(*z)


@_dispatch
def code(p: Parallel, xz: Parallel, z: Parallel, x, **kw_args):
    xz, z = zip(*[code(pi, xzi, zi, x, **kw_args) for (pi, xzi, zi) in zip(p, xz, z)])
    return Parallel(*xz), Parallel(*z)


@_dispatch
def code_track(p: Parallel, xz, z, x, h, **kw_args):
    xz, z, hs = zip(*[code_track(pi, xz, z, x, [], **kw_args) for pi in p])
    return Parallel(*xz), Parallel(*z), h + [Parallel(*hs)]


@_dispatch
def code_track(p: Parallel, xz, z: Parallel, x, h, **kw_args):
    xz, z, hs = zip(
        *[code_track(pi, xz, zi, x, [], **kw_args) for (pi, zi) in zip(p, z)]
    )
    return Parallel(*xz), Parallel(*z), h + [Parallel(*hs)]


@_dispatch
def code_track(p: Parallel, xz: Parallel, z: Parallel, x, h, **kw_args):
    xz, z, hs = zip(
        *[code_track(pi, xzi, zi, x, [], **kw_args) for (pi, xzi, zi) in zip(p, xz, z)]
    )
    return Parallel(*xz), Parallel(*z), h + [Parallel(*hs)]


@_dispatch
def recode(p: Parallel, xz, z, h, **kw_args):
    xz, z, _ = zip(*[recode(pi, xz, z, hi, **kw_args) for pi, hi in zip(p, h[0])])
    return Parallel(*xz), Parallel(*z), h[1:]


@_dispatch
def recode(p: Parallel, xz, z: Parallel, h, **kw_args):
    xz, z, _ = zip(
        *[recode(pi, xz, zi, hi, **kw_args) for (pi, zi, hi) in zip(p, z, h[0])]
    )
    return Parallel(*xz), Parallel(*z), h[1:]


@_dispatch
def recode(p: Parallel, xz: Parallel, z: Parallel, h, **kw_args):
    xz, z, _ = zip(
        *[
            recode(pi, xzi, zi, hi, **kw_args)
            for (pi, xzi, zi, hi) in zip(p, xz, z, h[0])
        ]
    )
    return Parallel(*xz), Parallel(*z), h[1:]


def broadcast_coder_over_parallel(coder_type):
    """Broadcast a coder over parallel encodings.

    Args:
        coder_type (type): Type of coder.
    """

    @_dispatch
    def code(p: coder_type, xz: Parallel, z: Parallel, x, **kw_args):
        xz, z = zip(*[code(p, xzi, zi, x, **kw_args) for (xzi, zi) in zip(xz, z)])
        return Parallel(*xz), Parallel(*z)
