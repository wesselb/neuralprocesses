import lab as B

from .util import nps, approx  # noqa


def test_discretisation(nps):
    disc = nps.Discretisation(points_per_unit=33, multiple=5, margin=0.05)

    x1 = B.linspace(nps.dtype, 0.1, 0.5, 10)[None, None, :]
    x2 = B.linspace(nps.dtype, 0.2, 0.6, 15)[None, None, :]

    grid = disc(x1, x2)

    # Check begin and start.
    assert B.min(grid) <= 0.1 - 0.05
    assert B.max(grid) >= 0.6 + 0.05

    # Check resolution.
    approx(grid[1:] - grid[:-1], 1 / 33, atol=1e-8)

    # Check that everything is on a global grid.
    approx(grid * 33, B.to_numpy(grid * 33).astype(int), atol=1e-8)

    # Check that overshoot is balanced.
    overshoot_left = (0.1 - 0.05) - B.min(grid)
    overshoot_right = B.max(grid) - (0.6 + 0.05)
    assert B.abs(overshoot_left - overshoot_right) <= 1 / 33
