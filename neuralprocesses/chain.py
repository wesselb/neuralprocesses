from matrix.util import indent

from . import _dispatch
from .util import register_module, is_framework_module

__all__ = ["Chain"]


@register_module
class Chain:
    """A chain of links.

    Args:
        *links (object): Links of the chain.

    Attributes:
        links (tuple): Links of the chain.
    """

    def __init__(self, *links):
        # Filter `None`s.
        links = tuple(filter(None, links))
        if any(is_framework_module(link) for link in links):
            self.links = self.nn.ModuleList(links)
        else:
            self.links = links

    def __call__(self, x):
        for link in self.links:
            x = link(x)
        return x

    def __getitem__(self, item):
        return self.links[item]

    def __len__(self):
        return len(self.links)

    def __iter__(self):
        return iter(self.links)

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


@_dispatch
def code_track(chain: Chain, xz, z, x, h, **kw_args):
    for link in chain:
        xz, z, h = code_track(link, xz, z, x, h, **kw_args)
    return xz, z, h


@_dispatch
def recode(chain: Chain, xz, z, h, **kw_args):
    for link in chain:
        xz, z, h = recode(link, xz, z, h, **kw_args)
    return xz, z, h
