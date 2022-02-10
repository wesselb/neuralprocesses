import pytest

import neuralprocesses as nps


def test_chain():
    c = nps.Chain(lambda x: x - 1)
    assert c(3) == 2

    # Check that the links of the chain are processed in the right order.
    c = nps.Chain(lambda x: x - 1, lambda x: x**2)
    assert c(3) == 4
    c = nps.Chain(lambda x: x**2, lambda x: x - 1)
    assert c(3) == 8
