import argparse
import logging
import pytest
import torch
import numpy as np

from .util import nps
import matplotlib.pyplot as plt
import neuralprocesses.ar.sampler as sp
import neuralprocesses.ar.trajectory as tj
import neuralprocesses.ar.viz as viz

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)

# Dummy generator
# dg.num_target.upper
# str(dg)
# dg.batch_size
# dg.num_context = UniformDiscrete(1, 1)
# dg.generate_batch()


@pytest.fixture(scope="session")
def density_loc(tmp_path_factory):
    fn = tmp_path_factory.mktemp("data") / "densities.hdf5"
    return fn


@pytest.fixture()
def tg():
    # tg = tj.GridGenerator(trajectory_length=5, min_x=-2, max_x=2)
    tg = tj.RandomGenerator(trajectory_length=5, min_x=-2, max_x=2)
    # tg = tj.RandomGenerator(trajectory_length=5, min_x=0.45, max_x=.55)
    return tg


def test_traj_gen(tg):
    seed = 0
    # batch size of 3
    x_context = np.array([0.0, 0.5, 0.1])
    traj = tg.generate(x_context, seed=seed)
    correct_traj = torch.Tensor([
        [[0., 1.]],
        [[1., 0.]],
        [[1., 0.]],
    ])
    torch.testing.assert_allclose(traj, correct_traj)


@pytest.fixture()
def dg():
    dg_kwargs = {
        "data": "sawtooth",
        "dim_x": 1,
        "dim_y": 1,
        "batch_size": 8,
        # is this doing anything right now?
    }
    dg = sp.get_generator(dg_kwargs, num_context=1, device="cpu")
    return dg


def test_xt_shape(ts):
    assert ts.xt.shape[0] == 6  # num context ranges * num_funcs_per context
    assert ts.xt.shape[-1] == 100  # number of target points


def test_generate_batch(ts):
    b = ts.generate_batch(
        batch_size=2,
        num_context=2,
    )
    assert b['xt'].shape == (2, 1, 100)  # func_per_context, 1, num_target


def make_sawtooth_wave(reps=2):
    xt = torch.linspace(-2, 2, 100)
    yt = (3 * (xt - 0.5)) % 1

    xt = xt.reshape(1, 1, -1).repeat((reps, 1, 1))
    yt = yt.reshape(1, 1, -1).repeat((reps, 1, 1))

    return xt, yt


def test_add1contexts(ts):
    context_size = 1
    cx = torch.Tensor([[[0.0]], [[1.0]]])
    cy = torch.Tensor([[[0.5]], [[0.5]]])

    func_context = [(cx, cy)]
    xt, yt = make_sawtooth_wave(2)  # num funcs per context
    ts._xt[0:2, :, :] = xt
    ts._yt[0:2, :, :] = yt

    assert func_context[0][0].shape == (2, 1, 1)
    # funcs_per_context, 1, context_size

    ss = sp.FunctionTrajectorySet(
        hdf5_loc=ts.density_loc,
        contexts=func_context,
        trajectory_generator=ts.trajectory_generator,
        group_name=f"{context_size}",
    )
    ss.create_samples(ts.model, ts.num_trajectories)


def test_add2contexts(ts):
    context_size = 2
    cx = torch.Tensor([[[0.0, 0.25]], [[1.0, 0.75]]])
    cy = torch.Tensor([[[0.5, 0.25]], [[0.5, 0.75]]])

    func_context = [(cx, cy)]
    xt, yt = make_sawtooth_wave(2)  # num funcs per context
    ts._xt[2:4, :, :] = xt
    ts._yt[2:4, :, :] = yt

    assert func_context[0][0].shape == (2, 1, 2)
    # funcs_per_context, 1, context_size
    ss = sp.FunctionTrajectorySet(
        hdf5_loc=ts.density_loc,
        contexts=func_context,
        trajectory_generator=ts.trajectory_generator,
        group_name=f"{context_size}",
    )
    ss.create_samples(ts.model, ts.num_trajectories)


def test_generate(ts: sp.TrajectorySet):
    test_add1contexts(ts)
    test_add2contexts(ts)
    dkwargs = {
        "start": -0.1,
        "end": 1.1,
        "step": 0.01  # this part is cheap to go dense
    }
    ts.create_density_grid("grid", density_kwargs=dkwargs)

    j = 4
    my_anim = viz.MyAnimator(ts.density_loc, 1, 0)
    my_anim.set_likelihoods()
    my_anim.set_frame_data(method="all_trajectory_lengths", frame_data=None)
    my_anim.set_densities(nlevels_min=500, quantile=0.95)
    my_anim.set_first_frame()
    # my_anim.animate(j)
    anim_loc = "../notebooks/animation.mp4"
    fps = 8
    my_anim.write_animation(str(anim_loc), fps)

    my_anim.figure
    plt.show()
    2 + 2


@pytest.fixture()
def ts(density_loc, dg, tg):
    num_functions_per_context_size = 2
    num_trajectories = 16
    context_range = (1, 2)
    device = "cpu"
    weights_loc = "../models/model-last.torch"
    model = sp.load_model(weights_loc, name="sawtooth")
    ts = sp.TrajectorySet(
        density_loc,
        model=model,
        # ^ not good practice to have outside dependencies in tests
        data_generator=dg,
        trajectory_generator=tg,
        num_functions_per_context_size=num_functions_per_context_size,
        num_trajectories=num_trajectories,
        context_range=context_range,
        device=device,
    )
    return ts


def test_calc_loglikelihood(nps):
    # 4 target points with trajectory length 2
    lh0 = np.array([
        [0.0, 0.5],
        [0.25, 0.75],
        [0.5, 0.5],
        [0.5, 0.1],
    ])
    expected_ll = sp.get_func_expected_ll(lh0)
    assert np.isclose(expected_ll, -3.97656152)
