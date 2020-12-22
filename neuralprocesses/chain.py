from . import _dispatch


__all__ = ["Chain"]


class Chain:
    def __init__(self, *links):
        self.links = links

    def __call__(self, x):
        for link in self.links:
            x = link(x)
        return x


@_dispatch()
def code():
    pass
