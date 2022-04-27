from plum import Tuple

from . import _dispatch

__all__ = ["Union"]


class Union:
    """A union of inputs selecting particular outputs.

    Args:
        *elements (tuple[object, int]): A tuple of inputs and integers where the integer
            selects the particular output.
    """

    @_dispatch
    def __init__(self, *elements: Tuple[object, int]):
        self.elements = elements
