import lab as B
import pytest

from .test_architectures import check_prediction
from .util import nps  # noqa


@pytest.mark.flaky(reruns=3)
def test_convgnp_auxiliary_variable(nps):
    model = nps.construct_convgnp(
        dim_x=2,
        dim_yc=(3, 1, 2),
        dim_aux_t=4,
        dim_yt=3,
        num_basis_functions=16,
        points_per_unit=16,
        likelihood="lowrank",
    )

    observed_data = (
        B.randn(nps.dtype, 16, 2, 10),
        B.randn(nps.dtype, 16, 3, 10),
    )
    aux_var1 = (
        B.randn(nps.dtype, 16, 2, 12),
        B.randn(nps.dtype, 16, 1, 12),
    )
    aux_var2 = (
        (B.randn(nps.dtype, 16, 1, 25), B.randn(nps.dtype, 16, 1, 35)),
        B.randn(nps.dtype, 16, 2, 25, 35),
    )
    aux_var_t = B.randn(nps.dtype, 16, 4, 15)
    pred = model(
        [observed_data, aux_var1, aux_var2],
        B.randn(nps.dtype, 16, 2, 15),
        aux_t=aux_var_t,
    )

    check_prediction(nps, pred, B.randn(nps.dtype, 16, 3, 15))
