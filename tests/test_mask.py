import lab as B
import pytest

from .test_architectures import generate_data
from .util import approx, nps  # noqa


@pytest.mark.flaky(reruns=3)
def test_convgnp_mask(nps):
    model = nps.construct_convgnp(
        num_basis_functions=16,
        points_per_unit=16,
        conv_arch="conv",
        conv_receptive_field=0.5,
        conv_layers=1,
        conv_channels=1,
        # Dividing by the density channel makes the forward very sensitive to the
        # numerics.
        divide_by_density=False,
    )
    xc, yc, xt, yt = generate_data(nps)

    # Predict without the final three points.
    pred = model(xc[:, :, :-3], yc[:, :, :-3], xt)
    # Predict using a mask instead.
    mask = B.to_numpy(B.ones(yc))  # Perform assignment in NumPy.
    mask[:, :, -3:] = 0
    mask = B.cast(B.dtype(yc), mask)
    pred_masked = model(xc, nps.Masked(yc, mask), xt)

    # Check that the two ways of doing it coincide.
    approx(pred.mean, pred_masked.mean)
    approx(pred.var, pred_masked.var)


@pytest.mark.parametrize("ns", [(10,), (0,), (10, 5), (10, 0), (0, 10), (15, 5, 10)])
@pytest.mark.parametrize("multiple", [1, 2, 3, 5])
def test_mask_contexts(nps, ns, multiple):
    x, y = nps.merge_contexts(
        *((B.randn(nps.dtype, 2, 3, n), B.randn(nps.dtype, 2, 4, n)) for n in ns),
        multiple=multiple
    )

    # Test that the output is of the right shape.
    if max(ns) == 0:
        assert B.shape(y.y, 2) == multiple
    else:
        assert B.shape(y.y, 2) == ((max(ns) - 1) // multiple + 1) * multiple

    # Test that the mask is right.
    mask = y.mask == 1  # Convert mask to booleans.
    assert B.all(B.take(B.flatten(y.y), B.flatten(mask)) != 0)
    assert B.all(B.take(B.flatten(y.y), B.flatten(~mask)) == 0)
