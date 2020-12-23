# from . import _dispatch


__all__ = ["Chain"]


class Chain:
    def __init__(self, *links):
        self.links = links

    def __call__(self, x):
        for link in self.links:
            x = link(x)
        return x


@_dispatch(Chain, B.Numeric, B.Numeric, B.Numeric)
def code(c, xz, z, x, **kw_args):
    for ci in c.links:
            xz, z = code(ci, xz, z, x, **kw_args)
    return xz, z




