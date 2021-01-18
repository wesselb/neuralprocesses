from . import _dispatch


__all__ = ["Chain"]


class Chain:
    """A chain of links.

    Args:
        *links (tuple): Links of the chain.
    """

    def __init__(self, *links):
        self.links = links

    def __call__(self, x):
        for link in self.links:
            x = link(x)
        return x


@_dispatch(Chain, object, object, object)
def code(chain, xz, z, x, **kw_args):
    for link in chain.links:
        xz, z = code(link, xz, z, x, **kw_args)
    return xz, z
