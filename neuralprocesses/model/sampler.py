import logging
import os
from pathlib import Path

import h5py
import torch

from neuralprocesses import torch as nps

LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def read_hdf5(hdf5_loc: Path):
    if not hdf5_loc.exists():
        raise FileNotFoundError(f"{hdf5_loc} does not exist")
    ss = SampleSet(hdf5_loc=hdf5_loc, contexts=None, ar_inputs=None)
    return ss


class SampleSet:
    def __init__(self, contexts, ar_inputs, hdf5_loc: Path, overwrite=False):
        self.hdf5_loc = hdf5_loc
        self.data = None

        if self.hdf5_loc.exists() and overwrite is False:
            LOG.warning("HDF5 file already exists. Using existing file.")
            LOG.warning("ar_inputs, and contexts will be ignored.")
        else:
            nar = ar_inputs.numpy().reshape(-1)
            cx_np = contexts[0][0].detach().numpy().reshape(-1)
            cy_np = contexts[0][1].detach().numpy().reshape(-1)
            with h5py.File(self.hdf5_loc, "w") as f:
                f.create_dataset("ar_inputs", data=nar)
                f.create_dataset("cx", data=cx_np)
                f.create_dataset("cy", data=cy_np)

    @property
    def ar_inputs(self):
        with h5py.File(self.hdf5_loc, "r") as f:
            np_ari = f["ar_inputs"][:]
            stored_ar_inputs = torch.Tensor(np_ari).reshape(1, 1, -1)
        return stored_ar_inputs

    @property
    def contexts(self):
        with h5py.File(self.hdf5_loc, "r") as f:
            cx = f["cx"][:].reshape(1, 1, -1)  # not sure if -1 should be for last index
            cy = f["cy"][:].reshape(1, 1, -1)
        fc = [(torch.Tensor(cx), torch.Tensor(cy))]
        return fc

    @property
    def sample_size(self):
        return self.ar_inputs.shape[-1]

    @property
    def y_traj(self):
        self.check_traj()
        with h5py.File(self.hdf5_loc, "r") as f:
            np_y_traj = f["y_traj"][:]
            y_traj = torch.Tensor(np_y_traj).reshape(-1, 1, 1, self.sample_size)
        return y_traj

    @property
    def x_traj(self):
        self.check_traj()
        return self.ar_inputs.repeat(self.y_traj.shape[0], 1, 1, 1)

    @property
    def traj(self):
        self.check_traj()
        return [self.x_traj, self.y_traj]

    def create_samples(self, model, n_samples):
        with h5py.File(self.hdf5_loc, "r") as f:
            if "y_traj" in f:
                LOG.warning("Samples already created. Skipping.")
                return

        LOG.info(f"Initial context: {self.contexts}")
        LOG.info(f"Generating {n_samples} trajectories of length {len(self.ar_inputs)}")

        traj = get_trajectories(model, self.ar_inputs, n_samples, self.contexts)
        y_traj = traj[1].detach().numpy().reshape(-1, self.sample_size)

        with h5py.File(self.hdf5_loc, "a") as f:
            f.create_dataset("y_traj", data=y_traj)

    def check_traj(self):
        with h5py.File(self.hdf5_loc, "r") as f:
            if "y_traj" not in f:
                raise Exception("Samples not yet created.")


def get_trajectories(model, xi, n_mixtures, contexts):
    # Generate AR trajectories from the fixed selections in the domain.
    ag = nps.AggregateInput((xi, 0))
    mean, var, noiseless_samples, noisy_samples = nps.ar_predict(
        model, contexts, ag, num_samples=n_mixtures, order="given"
    )
    traj = (xi, noisy_samples.elements[0])
    num_samples = traj[1].shape[0]
    rep_x = traj[0].repeat(num_samples, 1, 1, 1)
    traj = list(tuple([rep_x, traj[1]]))
    return traj
