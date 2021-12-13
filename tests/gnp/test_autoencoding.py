import lab as B
import pytest

import neuralprocesses.gnp as gnp

# noinspection PyUnresolvedReferences
from .util import context_set, target_set


@pytest.fixture()
def disc():
    return gnp.Discretisation1d(points_per_unit=32, multiple=4, margin=0.1)


def test_autoencoding_1d(disc, context_set, target_set):
    enc = gnp.SetConv1dEncoder(disc)
    dec = gnp.SetConv1dDecoder(disc)

    xz, z = enc.forward(*context_set, target_set[0])
    xz, z = dec.forward(xz, z, target_set[0])

    target_shape = (
        B.shape(target_set[1])[0],
        B.shape(target_set[1])[1],
        B.shape(target_set[1])[2] + 1,
    )
    assert B.shape(z) == target_shape


def test_autoencoding_2d(disc, context_set, target_set):
    enc = gnp.SetConv1dPDEncoder(disc)
    dec = gnp.SetConv1dPDDecoder(disc)

    xz, z = enc.forward(*context_set, target_set[0])
    xz, z = dec.forward(xz, z, target_set[0])

    target_shape = (
        B.shape(target_set[1])[0],
        B.shape(target_set[1])[2] + 2,
        B.shape(target_set[1])[1],
        B.shape(target_set[1])[1],
    )
    assert B.shape(z) == target_shape
