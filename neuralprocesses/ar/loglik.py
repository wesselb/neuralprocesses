import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from neuralprocesses.ar.sampler import Groups, Datasets, get_func_expected_ll

logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def check_vals(val, max_val):
    if val > max_val:
        raise ValueError(f"{val} is greater than {max_val}")


class LikelihoodCalculator:

    def __init__(self, density_loc: Path):
        self.density_loc = density_loc
        with h5py.File(self.density_loc, "r") as f:
            mds = f[Groups.MARGINAL_DENSITIES.value]
            keys = [k for k in mds.keys()]
            k = keys[0]
            lls = mds[k][Datasets.LIKELIHOODS.value]

            self.keys = keys
            self.num_targets = lls.shape[0]
            self.num_trajectories = lls.shape[1]
            self.num_density_evaluation_points = lls.shape[2] # <- this should be 1
            self.trajectory_length = lls.shape[3]
            self.num_functions = len(keys)

        context_sizes = []
        func_inds = []
        for k in keys:
            cs_str, func_ind_str = k.split("|")
            cs = int(cs_str)
            func_ind = int(func_ind_str)
            context_sizes.append(cs)
            func_inds.append(func_ind)

        self.context_sizes = np.array(context_sizes)
        self.num_functions_per_context_size = np.max(func_inds)

        if self.num_density_evaluation_points != 1:
            nde = self.num_density_evaluation_points
            raise ValueError(
                f"Expected density eval points to be 1, got {nde}")

    def grid_lls(self, context_size=None):
        if context_size not in self.context_sizes:
            raise ValueError(f"Context size {context_size} not in {self.context_sizes}")
        grid = np.zeros((self.num_trajectories, self.trajectory_length))
        for nt in tqdm(np.arange(self.num_trajectories)):
            for tl in np.arange(self.trajectory_length):
                grid[nt, tl] = self.calc_loglikelihood(nt, tl, context_size)
        return grid

    def calc_loglikelihood(self, num_trajectories, trajectory_length, context_size=None):
        check_vals(num_trajectories, self.num_trajectories)
        check_vals(trajectory_length, self.trajectory_length)

        func_loglikelihoods = np.zeros((self.num_functions))
        for i, k in enumerate(self.keys):
            cs_str, _ = k.split("|")
            cs = int(cs_str)

            if context_size is not None:
                if cs != context_size:
                    continue

            with h5py.File(self.density_loc, "r") as f:
                grp = f[Groups.MARGINAL_DENSITIES.value][k]
                lh = grp[Datasets.LIKELIHOODS.value]
                # the first of the eval points is the true y target
                # that is why we evaluate the likelihood there
                lh0 = lh[:, :num_trajectories, 0, trajectory_length][:]
                lh0[lh0 == 0] = 1e-10  # replace 0 likelihoods with a small value
                expected_ll = get_func_expected_ll(lh0)
                # Just using whatever order keys come in
                # Should be fine because order should not matter, since we take the mean
                func_loglikelihoods[i] = expected_ll
        return func_loglikelihoods.mean()


def main(msg):
    LOG.info(f'{msg}')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('i', help='input txt file')
    args = parser.parse_args()
    main(args.i)
