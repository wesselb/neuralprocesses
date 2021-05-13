import lab as B

import neuralprocesses.gnp as gnp
from ..util import approx


def test_discretisation():
    disc = gnp.Discretisation1d(points_per_unit=33, multiple=5, margin=0.05)

    x1 = B.linspace(0.1, 0.5, 10)
    x2 = B.linspace(0.2, 0.6, 15)

    grid = disc(x1, x2)

    # Check begin and start.
    assert min(grid) <= 0.1 - 0.05
    assert max(grid) >= 0.6 + 0.05

    # Check resolution.
    approx(grid[1:] - grid[:-1], 1 / 33, atol=1e-8)

    # Check that everything is on a global grid.
    approx(grid * 33, (grid * 33).astype(int), atol=1e-8)

    # Check that overshoot is balanced.
    overshoot_left = (0.1 - 0.05) - min(grid)
    overshoot_right = max(grid) - (0.6 + 0.05)
    assert abs(overshoot_left - overshoot_right) <= 1 / 33
