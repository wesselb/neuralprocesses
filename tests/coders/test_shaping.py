import pytest
from plum import Dispatcher
import neuralprocesses

from ..util import nps  # noqa

_dispatch = Dispatcher()


@_dispatch
def _to_tuple(x):
    return x


@_dispatch
def _to_tuple(x: tuple):
    return tuple(_to_tuple(xi) for xi in x)


@_dispatch
def _to_tuple(p: neuralprocesses.Parallel):
    return tuple(_to_tuple(pi) for pi in p)


def test_restructure_parallel(nps):
    reorg = nps.RestructureParallel((0, (1, 2)), (0, (2,), 1))

    res = nps.code(
        reorg,
        nps.Parallel("x1", nps.Parallel("x2", "x3")),
        nps.Parallel("y1", nps.Parallel("y2", "y3")),
        None,
        root=True,
    )
    assert _to_tuple(res) == (("x1", ("x3",), "x2"), ("y1", ("y3",), "y2"))

    # Check that the structure must be right.
    with pytest.raises(RuntimeError, match="Parallel does not match structure."):
        nps.code(
            reorg,
            nps.Parallel("x1", "x2", nps.Parallel("x3")),
            nps.Parallel("y1", nps.Parallel("y2", "y3")),
            None,
            root=True,
        )
