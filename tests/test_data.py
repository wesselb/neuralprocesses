import lab as B
import pytest

from .util import nps  # noqa


@pytest.mark.parametrize("dim_x", [1, 2])
@pytest.mark.parametrize("dim_y", [1, 2])
def test_predefined_gens(nps, dim_x, dim_y):
    gens = nps.construct_predefined_gens(nps.dtype, dim_x=dim_x, dim_y=dim_y)
    for name, gen in gens.items():
        for _ in range(10):
            batch = gen.generate_batch()

            assert B.shape(batch["xc"], 0) == gen.batch_size
            assert B.shape(batch["yc"], 0) == gen.batch_size
            assert B.shape(batch["xt"], 0) == gen.batch_size
            assert B.shape(batch["yt"], 0) == gen.batch_size

            assert B.shape(batch["xc"], 1) == dim_x
            assert B.shape(batch["yc"], 1) == dim_y
            assert B.shape(batch["xt"], 1) == dim_x
            assert B.shape(batch["yt"], 1) == dim_y

            if name != "mixture":
                assert B.shape(batch["xc"], 2) == B.shape(batch["yc"], 2)
                assert gen.num_context_points[0] <= B.shape(batch["xc"], 2)
                assert B.shape(batch["xc"], 2) <= gen.num_context_points[1]

                assert B.shape(batch["xt"], 2) == B.shape(batch["yt"], 2)
                assert gen.num_target_points[0] <= B.shape(batch["xt"], 2)
                assert B.shape(batch["xt"], 2) <= gen.num_target_points[1]
