# from . import _dispatch


__all__ = ["Chain"]


class Chain:
    def __init__(self, *links):
        self.links = links

    def __call__(self, x):
        for link in self.links:
            x = link(x)
        return x

class Parallel:
    def __init__(self, *elements):
        self.elements


@_dispatch(Chain, B.Numeric, B.Numeric, B.Numeric)
def code(c, xz, z, x, **kw_args):
    for ci in c.links:
        xz, z = code(ci, xz, z, x, **kw_args)
    return xz, z

@_dispatch(Parallel, B.Numeric, B.Numeric, B.Numeric)
def code(p, xz, z, x, **kw_args):
    xz, z = zip([code(pi, xz, z, x, **kw_args) for pi in p.elements])
    return Parallel(xz), Parallel(z)

@_dispatch(Parallel, B.Numeric, Parallel, B.Numeric)
def code(p, xz, z, x, **kw_args):
    xz, z = zip([code(pi, xz, zi, x, **kw_args) for (pi, zi) in zip(p.elements, z.elements)])
    return Parallel(xz), Parallel(z)

@_dispatch(Parallel, Parallel, Parallel, B.Numeric)
def code(p, xz, z, x, **kw_args):
    xz, z = zip([code(pi, xzi, zi, x, **kw_args) for (pi, xzi, zi) 
        in zip(p.elements, zi.elements,  z.elements)])
    return Parallel(xz), Parallel(z)

