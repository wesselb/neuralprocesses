from . import _dispatch
from matrix.util import indent


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

    def __iter__(self):
        return iter(self.links)

    def __getitem__(self, item):
        return self.links[item]

    def __repr__(self):
        return (
            "Chain(\n"
            + "".join([indent(repr(e).strip(), " " * 4) + ",\n" for e in self])
            + ")\n"
        )


@_dispatch
def code(chain: Chain, xz, z, x, **kw_args):
    for link in chain.links:
        xz, z = code(link, xz, z, x, **kw_args)
    return xz, z
