import lab as B
import pytest

from .test_architectures import generate_data
from .util import nps, approx  # noqa


@pytest.mark.flaky(reruns=3)
def test_convgnp_mask(nps):
    model = nps.construct_convgnp(
        num_basis_functions=16,
        points_per_unit=16,
        conv_arch="dws",
        dws_receptive_field=0.5,
        dws_layers=1,
        dws_channels=1,
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
