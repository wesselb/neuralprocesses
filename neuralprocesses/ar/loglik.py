import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import logging
import h5py
import numpy as np
from tqdm import tqdm
from pathlib import Path
from neuralprocesses.ar.sampler import Groups, Datasets, get_func_expected_ll
from neuralprocesses.ar.viz import make_heatmap

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
            lls = mds[k][Datasets.MEANS.value]

            self.keys = keys
            self.num_targets = lls.shape[1]
            self.num_trajectories = lls.shape[0]
            self.num_density_evaluation_points = 1
            self.trajectory_length = lls.shape[-1]
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
        if context_size is not None:
            if context_size not in self.context_sizes:
                raise ValueError(f"Context size {context_size} not in {self.context_sizes}")
        grid = np.zeros((self.num_trajectories, self.trajectory_length, self.num_functions))
        for nt in tqdm(np.arange(self.num_trajectories)):
            for tl in np.arange(self.trajectory_length):
                if nt == 0:
                    grid[nt, tl, :] = np.nan
                else:
                    grid[nt, tl, :] = self.calc_loglikelihood(nt, tl, context_size)
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
                lh = grp[Datasets.LOG_LIKELIHOODS.value]
                # the first of the eval points is the true y target
                # that is why we evaluate the likelihood there
                lh0 = lh[:, :num_trajectories, 0, trajectory_length][:]
                # if lh0[lh0 == 0].shape[0] > 0:
                #     # small_prob = min(lh0[lh0 != 0].min() / 10, 1e-16)
                #     nz_min = lh0[lh0 != 0].min()
                #     lh0[lh0 == 0] = nz_min
                # replace 0 likelihoods with a small value
                expected_ll = get_func_expected_ll(lh0)
                # Just using whatever order keys come in
                # Should be fine because order should not matter, since we take the mean
                func_loglikelihoods[i] = expected_ll
        return func_loglikelihoods


def main(model_dir: Path, show_fig=False):
    density_loc = model_dir / "densities.hdf5"

    if not model_dir.exists():
        raise ValueError(f"{model_dir} does not exist")
    if not density_loc.exists():
        raise ValueError(f"{density_loc} does not exist")

    lc = LikelihoodCalculator(density_loc)
    all_lls = lc.grid_lls(context_size=None)
    all_lls_func_means = np.mean(all_lls, axis=-1)
    fig = make_heatmap(all_lls_func_means)

    np.save(str(model_dir / "log_likelihoods.npy"), all_lls)
    fig.savefig(str(model_dir / "log_likelihoods.png"))

    if show_fig:
        plt.show()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Say hello')
    parser.add_argument('in_dir', help='directory where densities reside', type=Path)
    parser.add_argument('--show-plot', help='show figure', action='store_true')
    parser.set_defaults(show_figure=True)
    args = parser.parse_args()
    main(args.in_dir, args.show_plot)
