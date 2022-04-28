import lab as B
import pytest

from .util import nps  # noqa


@pytest.mark.parametrize("dim_x", [1, 2])
@pytest.mark.parametrize("dim_y", [1, 2])
def test_predefined_gens(nps, dim_x, dim_y):
    gens = nps.construct_predefined_gens(
        nps.dtype,
        x_range_context=(-2, 2),
        x_range_target=(2, 6),
        dim_x=dim_x,
        dim_y=dim_y,
    )
    if dim_x == 1 and dim_y == 2:
        gens["predprey"] = nps.PredPreyGenerator(
            nps.dtype,
            x_ranges_context=((0, 50),),
            x_ranges_target=((50, 100),),
            dim_y=2,
        )
    for name, gen in gens.items():
        for _ in range(10):
            batch = gen.generate_batch()

            # Check first dimension of shape.
            assert B.shape(batch["xc"], 0) == gen.batch_size
            assert B.shape(batch["yc"], 0) == gen.batch_size
            assert B.shape(batch["xt"], 0) == gen.batch_size
            assert B.shape(batch["yt"], 0) == gen.batch_size

            # Check second dimension of shape.
            assert B.shape(batch["xc"], 1) == dim_x
            assert B.shape(batch["yc"], 1) == dim_y
            assert B.shape(batch["xt"], 1) == dim_x
            assert B.shape(batch["yt"], 1) == dim_y

            # Check third dimension of shape.
            if name != "mixture":
                assert B.shape(batch["xc"], 2) == B.shape(batch["yc"], 2)
                assert gen.num_context_points[0] <= B.shape(batch["xc"], 2)
                assert B.shape(batch["xc"], 2) <= gen.num_context_points[1]

                assert B.shape(batch["xt"], 2) == B.shape(batch["yt"], 2)
                assert gen.num_target_points[0] <= B.shape(batch["xt"], 2)
                assert B.shape(batch["xt"], 2) <= gen.num_target_points[1]

                # Check the location of the inputs.
                for d in range(dim_x):
                    # Convert the limits to the right data type before comparing!
                    lower = B.cast(nps.dtype, gen.x_ranges_context[0][d])
                    upper = B.cast(nps.dtype, gen.x_ranges_context[1][d])
                    assert B.all(lower <= batch["xc"][:, d, :])
                    assert B.all(batch["xc"][:, d, :] <= upper)
                    
                    lower = B.cast(nps.dtype, gen.x_ranges_target[0][d])
                    upper = B.cast(nps.dtype, gen.x_ranges_target[1][d])
                    assert B.all(lower <= batch["xt"][:, d, :])
                    assert B.all(batch["xt"][:, d, :] <= upper)
