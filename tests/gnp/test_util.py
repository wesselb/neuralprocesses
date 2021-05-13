import neuralprocesses.gnp as gnp


def test_num_params():
    model = gnp.GNP()
    assert isinstance(gnp.num_params(model), int)
    assert gnp.num_params(model) > 0
