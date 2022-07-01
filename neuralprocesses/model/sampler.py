import logging
import os
from pathlib import Path

import h5py
import torch
from tqdm import tqdm

from neuralprocesses import torch as nps

LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def read_hdf5(hdf5_loc: Path):
    if not hdf5_loc.exists():
        raise FileNotFoundError(f"{hdf5_loc} does not exist")
    ss = SampleSet(hdf5_loc=hdf5_loc)
    return ss


class SampleSet:
    def __init__(self, hdf5_loc: Path, contexts=None, gen=None, overwrite=False):
        self.hdf5_loc = hdf5_loc
        self.data = None
        self.gen = gen

        if self.hdf5_loc.exists() and overwrite is False:
            if contexts is not None:
                LOG.warning(f"HDF5 file already exists. Using existing file at {self.hdf5_loc}.")
                LOG.warning("ar_inputs, and contexts will be ignored.")
            else:
                LOG.warning(f"Loading from {self.hdf5_loc}")
        else:
            cx_np = contexts[0][0].detach().numpy().reshape(-1)
            cy_np = contexts[0][1].detach().numpy().reshape(-1)
            with h5py.File(self.hdf5_loc, "w") as f:
                f.attrs["trajectory_generator"] = str(gen)
                f.attrs["sample_size"] = gen.trajectory_length
                f.create_dataset("cx", data=cx_np)
                f.create_dataset("cy", data=cy_np)

    @property
    def trajectory_generator(self):
        with h5py.File(self.hdf5_loc, "r") as f:
            traj_gen = f.attrs["trajectory_generator"]
        return traj_gen

    @property
    def sample_size(self):
        with h5py.File(self.hdf5_loc, "r") as f:
            sample_size = int(f.attrs["sample_size"])
        return sample_size


    @property
    def contexts(self):
        with h5py.File(self.hdf5_loc, "r") as f:
            cx = f["cx"][:].reshape(1, 1, -1)  # not sure if -1 should be for last index
            cy = f["cy"][:].reshape(1, 1, -1)
        fc = [(torch.Tensor(cx), torch.Tensor(cy))]
        return fc

    def check_traj(self):
        with h5py.File(self.hdf5_loc, "r") as f:
            if "y_traj" not in f:
                raise Exception("Samples not yet created.")

    def get_traj(self, dset_name):
        self.check_traj()
        with h5py.File(self.hdf5_loc, "r") as f:
            np_dim_traj = f[dset_name][:]
            dim_traj = torch.Tensor(np_dim_traj).reshape(-1, 1, 1, self.sample_size)
        return dim_traj

    @property
    def y_traj(self):
        y_traj = self.get_traj("y_traj")
        return y_traj

    @property
    def x_traj(self):
        x_traj = self.get_traj("x_traj")
        return x_traj

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
        LOG.info(f"Generating {n_samples} trajectories of length {self.sample_size}")

        x_traj = torch.zeros(n_samples, 1, 1, self.sample_size)
        y_traj = torch.zeros(n_samples, 1, 1, self.sample_size)
        inner_samples = 1  # TODO: could tweak this for more speed, but less stochasticity
        for i in tqdm(range(n_samples)):
            # TODO: assess whether this is fast enough, can I do it in parallel somehow?
            x_traj0 = self.gen.generate()
            y_traj0 = get_trajectories(model, x_traj0, inner_samples, self.contexts)
            x_traj[i, :, :, :] = x_traj0
            y_traj[i, :, :, :] = y_traj0

        with h5py.File(self.hdf5_loc, "a") as f:
            f.create_dataset("x_traj", data=x_traj.detach().numpy())

        with h5py.File(self.hdf5_loc, "a") as f:
            f.create_dataset("y_traj", data=y_traj.detach().numpy())


def get_trajectories(model, xi, n_mixtures, contexts):
    # Generate AR trajectories from the fixed selections in the domain.
    ag = nps.AggregateInput((xi, 0))
    mean, var, noiseless_samples, noisy_samples = nps.ar_predict(
        model, contexts, ag, num_samples=n_mixtures, order="given"
    )
    y_traj = noisy_samples.elements[0]
    # traj = (xi, noisy_samples.elements[0])
    # num_samples = traj[1].shape[0]
    # rep_x = traj[0].repeat(num_samples, 1, 1, 1)
    # traj = list(tuple([rep_x, traj[1]]))
    return y_traj
