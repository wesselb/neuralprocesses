from .util import nps, generate_data  # noqa


def test_num_params(nps):
    model = nps.construct_gnp()
    model(*generate_data(nps)[:3])  # Run forward to initialise parameters.
    assert isinstance(nps.num_params(model), int)
    assert nps.num_params(model) > 0
