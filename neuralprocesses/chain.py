from matrix.util import indent

from . import _dispatch
from .util import register_module

__all__ = ["Chain"]


@register_module
class Chain:
    """A chain of links.

    Args:
        *links (tuple): Links of the chain.
    """

    def __init__(self, *links):
        # Filter `None`s.
        links = tuple(filter(None, links))
        try:
            self.links = self.nn.ModuleList(links)
        except AttributeError:
            self.links = links

    def __call__(self, x):
        for link in self.links:
            x = link(x)
        return x

    def __iter__(self):
        return iter(self.links)

    def __getitem__(self, item):
        return self.links[item]

    def __str__(self):
        return repr(self)

    def __repr__(self):
        return (
            "Chain(\n"
            + "".join([indent(repr(e).strip(), " " * 4) + ",\n" for e in self])
            + ")"
        )


@_dispatch
def code(chain: Chain, xz, z, x, **kw_args):
    for link in chain:
        xz, z = code(link, xz, z, x, **kw_args)
    return xz, z
