import logging
from datetime import datetime
import argparse
from pathlib import Path
from tqdm import tqdm

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from time import time
import seaborn as sns
from scipy.stats import norm
import yaml
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

import scripts.ar_marginal as sam
from neuralprocesses.ar.loglik import get_func_expected_ll
from neuralprocesses.ar.enums import Groups, Datasets
import neuralprocesses.ar.utils as utils
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 300
# plt.rcParams.update({"figure.figsize": (14, 6)})


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def get_function_predictive_params(
        density_loc,
        context_size,
        function_index
):
    """
    Load the means and variances for all trajectory length and number of trajectories
    for the given function (as specific by context size and index).

    Args:
        density_loc: The location of the density data
        context_size: The context size of the function
        function_index: The index of the function

    Returns:
        A tuple of the means and variances, x target locations, and true y values
            at those locations.
    """
    with h5py.File(density_loc, "r") as f:
        mds = f[Groups.MARGINAL_DENSITIES.value]
        grp = mds[f"{context_size}|{function_index}"]
        xt = grp["x_targets"][:].reshape(-1)
        yt = grp["y_density_evaluation_points"][:, 0]
        # ^ first index contains true targets
        means = grp[Datasets.MEANS.value][:]
        variances = grp[Datasets.VARIANCES.value][:]
    return means, variances, xt, yt


def get_function_density(
        means,
        variances,
        xt,
        number_of_trajectories,
        trajectory_length,
        density_eval_locations,
):
    """
    Get the density for each target point for the provided function parameters

    Args:
        means: The means of the predictive model for this function
        variances:  The variances of the predictive model for this function
        xt:  The target points
        number_of_trajectories: The number of trajectories
        trajectory_length: The length of each trajectory
        density_eval_locations: The locations to evaluate the density at

    Returns:
        An array of densities for each target point with shape:
            (number of density evaluation points, number of targets)
    """
    nt = number_of_trajectories
    tl = trajectory_length

    param_m0 = means[:nt, :, :, :, tl]
    param_v0 = variances[:nt, :, :, :, tl]

    gmms_per_target = get_gmms_per_target(
        param_m0,
        param_v0,
        xt,
        nt,
        density_eval_locations,
    )
    return gmms_per_target


def get_gmms_per_target(param_m0, param_v0, xt, nt, density_eval_locations):
    """
    Get the GMMs density for each target point.

    Args:
        param_m0: The means of the predictive model
        param_v0: The variances of the predictive model
        xt: The target points
        nt: The number of trajectories
        density_eval_locations: The locations to evaluate the density at

    Returns:
        An array of GMMs for each target point with shape:
            (number of density evaluation points, number of targets)
    """
    num_density_eval_points = len(density_eval_locations)

    gmms_per_target = np.zeros((num_density_eval_points, xt.shape[0]))
    pbar = tqdm(enumerate(xt), total=xt.shape[0])
    for i, xt0 in pbar:
        gmm_components = np.zeros((nt, num_density_eval_points))
        for nt0 in range(nt):
            d = get_gmm_components(
                param_m0,
                param_v0,
                nt0,
                i,
                density_eval_locations,
            )
            gmm_components[nt0, :] = d
        gmm_d = gmm_components.mean(axis=0)
        gmms_per_target[:, i] = gmm_d
    return gmms_per_target


def get_gmm_components(
  param_m0, param_v0, nt0, i, density_eval_locations
):
    """
    Get the GMM components for the density.

    Args:
        param_m0: means of the predictive model
        param_v0: variances of the predictive model
        nt0: the current number of trajectories
        i: the current target point index
        density_eval_locations: the locations to evaluate the density at

    Returns:
        A GMM component evaluated at each density evaluation point
    """
    component_m0 = param_m0[nt0, i, :, :]
    component_v0 = param_v0[nt0, i, :, :]
    d = norm.pdf(x=density_eval_locations, loc=component_m0, scale=np.sqrt(component_v0))
    return d


class MyAnimator:
    def __init__(self, density_loc, num_contexts, func_ind, eval_points):
        self.density_loc = density_loc
        self.num_contexts = num_contexts
        self.func_ind = func_ind

        self.eval_points = eval_points

        self.lh = None
        self.xt = None
        self.true_y_targets = None
        self.cx = None
        self.cy = None
        self.total_num_trajectories = None
        self.total_trajectory_len = None

        self.frame_data = None

        self.densities = None
        self.targets = None
        self.density_eval_locations = None

        self.means = None
        self.variances = None

        self.vmax = None
        self.nlevels = None
        self.norm = None

        self.figure = None
        self.ax = None
        self.cont = None

    def set_densities(self):
        densities = []
        if self.frame_data is None:
            raise Exception("Frame data not set")
        for frame_index in range(len(self.frame_data)):
            num_traj0, traj_len0 = self.frame_data[frame_index]
            od, targets, density_eval_locations = self.get_density_grid(
                num_trajectories=num_traj0,
                trajectory_length=traj_len0,
            )
            densities.append(od)
        self.density_eval_locations = density_eval_locations
        self.targets = targets # should be the same each time
        self.densities = np.stack(densities)

    def set_contour_prefs(self, nlevels_min=100, max_quantile=0.95, min_quantile=0.05, norm=None):
        vmax = np.nanquantile(self.densities, max_quantile)
        vmin = np.nanquantile(self.densities, min_quantile)
        if norm is None:
            self.norm = Normalize(vmin=vmin, vmax=vmax)
        else:
            self.norm = norm
        self.nlevels = min(nlevels_min, int(self.density_eval_locations.shape[0] / 2))
        LOG.info(f"Using {self.nlevels} levels with norm={self.norm}")
        LOG.info(f"Density shape: {self.densities.shape}")

    def set_frame_data(self, method="all_trajectory_lengths", frame_data=None):
        # First column is number of trajectories, second column is trajectory length
        if method == "all_trajectory_lengths":
            frame_data = np.empty((self.total_trajectory_len, 2), dtype=int)
            frame_data[:, 0] = self.total_num_trajectories
            frame_data[:, 1] = np.arange(self.total_trajectory_len)
        elif method == "all_num_trajectories":
            frame_data = np.empty((self.total_num_trajectories - 1, 2))
            frame_data[:, 0] = np.arange(self.total_num_trajectories)[
                1:
            ]  # skip first bc nan
            frame_data[:, 1] = self.total_trajectory_len - 1
        elif method == "provided":
            if frame_data is not None:
                frame_data = np.array(frame_data)  # coerce to numpy if list
            else:
                raise Exception("No frame data provided.")
        else:
            raise Exception(f"Unknown method: {method}")
        frame_data = frame_data.astype(int)
        self.frame_data = frame_data

    def set_likelihoods(self):
        with h5py.File(self.density_loc, "a") as f:
            gname = f"{self.num_contexts}|{self.func_ind}"
            grp = f[Groups.MARGINAL_DENSITIES.value][gname]
            tgrp = f[Groups.TRAJECTORIES.value]
            t0 = tgrp[f"{self.num_contexts}"]

            lh = grp[Datasets.LOG_LIKELIHOODS.value][:]
            xt = grp["x_targets"][:]
            eval_points = grp["y_density_evaluation_points"][:]
            cx = t0["cx"][self.func_ind, :]
            cy = t0["cy"][self.func_ind, :]

        self.lh = lh
        self.xt = xt
        self.true_y_targets = eval_points[:, 0]
        self.cx = cx
        self.cy = cy
        self.total_num_trajectories = lh.shape[1]
        self.total_trajectory_len = lh.shape[-1]
        LOG.info("Adding true y targets and y contexts to evaluation points.")
        self.eval_points = np.concatenate([
            self.true_y_targets,
            self.eval_points,
            cy.reshape(-1),
        ])

        self.means, self.variances, _, _ = get_function_predictive_params(
            self.density_loc,
            self.num_contexts,
            self.func_ind,
        )

    def get_density_grid(self, num_trajectories, trajectory_length):
        # density = self.lh[:, :num_trajectories, :, trajectory_length]
        gmm = get_function_density(
            self.means,
            self.variances,
            self.xt,
            num_trajectories,
            trajectory_length,
            self.eval_points,
        )
        # gmm = density.mean(axis=1).T
        x_order = np.argsort(self.xt, axis=0).reshape(-1)
        y_order = np.argsort(-self.eval_points, axis=0).reshape(-1)
        od = gmm[:, x_order][y_order, :]

        targets = self.xt[x_order].reshape(-1)
        density_eval_locations = self.eval_points[y_order]
        return od, targets, density_eval_locations

    def set_first_frame(self, target_model="scatter"):
        num_traj0, traj_len0 = self.frame_data[0]
        od = self.densities[0]
        targets = self.targets
        density_eval_locations = self.density_eval_locations
        LOG.info(
            f"Trajectory length: {traj_len0} | Number of trajectories: {num_traj0}"
        )

        self.figure, self.ax = plt.subplots(figsize=(20, 5))
        self.ax.set_xlim(targets.min(), targets.max())
        self.ax.set_ylim(density_eval_locations.min(), density_eval_locations.max())

        x = targets
        y = density_eval_locations
        self.cont = plt.contourf(x, y, od, self.nlevels, norm=self.norm, zorder=1)
        # first image on screen
        plt.scatter(
            self.cx, self.cy, s=150, marker="+", color="red", label="contexts", zorder=3
        )
        target_order = np.argsort(self.xt, axis=0).reshape(-1)
        xt_ordered = self.xt[target_order]
        yt_ordered = self.true_y_targets[target_order]
        plt.plot(
            xt_ordered,
            yt_ordered,
            color="black",
            label="targets",
            zorder=2,
            alpha=0.5
        )
        # nt, tl = my_anim.frame_data[j]
        lh0 = self.lh[:, :num_traj0, 0, traj_len0]
        ll = get_func_expected_ll(lh0)
        t0 = f"Densities with {num_traj0} trajectories of length {traj_len0} | L: {ll:.2f}"
        plt.title(t0)
        plt.legend()

    def animate(self, frame_index):
        num_traj0, traj_len0 = self.frame_data[frame_index]
        od = self.densities[frame_index]
        targets = self.targets
        density_eval_locations = self.density_eval_locations
        LOG.info(
            f"Trajectory length: {traj_len0} | Number of trajectories: {num_traj0}"
        )
        num_traj0, traj_len0 = self.frame_data[frame_index]
        if self.cont is None:
            raise Exception("First frame not set")
        for c in self.cont.collections:
            c.remove()  # removes only the contours, leaves the rest intact
        # There is a lot duplicate code in here
        cont = plt.contourf(
            targets,
            density_eval_locations,
            od,
            self.nlevels,
            cmap="viridis",
            # vmin=self.vmin,
            # vmax=self.vmax,
            norm=self.norm,
            zorder=1,
        )
        self.cont = cont
        lh0 = self.lh[:, :num_traj0, 0, traj_len0]
        ll = get_func_expected_ll(lh0)
        t0 = f"Densities with {num_traj0} trajectories of length {traj_len0} | L: {ll:.2f}"
        plt.title(t0)
        # plt.title(f"Densities with {num_traj0} trajectories of length {traj_len0}")
        return cont

    def write_animation(self, anim_loc, fps):
        anim = animation.FuncAnimation(
            self.figure,
            self.animate,
            frames=self.frame_data.shape[0],
            repeat=False,
        )
        anim.save(str(anim_loc), writer=animation.FFMpegWriter(fps=fps))


def setup(active_config_dir, model_dir):
    in_config = utils.make_menu(active_config_dir)
    with open(in_config, "r") as f0:
        orig_config = yaml.safe_load(f0)
    config = utils.clean_config(orig_config)

    if "experiment" in config:
        exp_name = config["experiment"]
    else:
        exp_name = in_config.stem
    exp_dir = Path(model_dir) / exp_name

    if "animations" not in config:
        raise Exception("No animations section in config")
    print(yaml.dump(config["animations"]))
    return config, exp_dir


def make_animations(config, exp_dir, only_first_frame=False):
    pbar = tqdm(config["animations"])
    for ac in pbar:
        cs = ac['num_contexts']
        func_ind = ac['func_ind']
        pbar.set_description(f"Context Size: {cs} | Function: {func_ind}")
        make_animation(exp_dir, ac, only_first_frame)
    LOG.info(f"Saved to \"{exp_dir}\".")


def main(active_config_dir, model_dir: Path, only_first_frame: bool = False):
    config, exp_dir = setup(active_config_dir, model_dir)
    make_animations(config, exp_dir, only_first_frame)


def make_animation(density_dir, config, only_first_frame=False):
    func_ind = config["func_ind"]
    num_contexts = config["num_contexts"]

    frame_data_method = config["frame_data_method"]
    frame_data = config["frame_data"]

    nlevels_min = config["nlevels_min"]
    max_quantile = config["max_quantile"]
    min_quantile = config["min_quantile"]
    fps = config["fps"]

    anim_dir = density_dir / "animations"
    anim_dir.mkdir(exist_ok=True)
    density_loc = density_dir / "densities.hdf5"

    LOG.info(f"Loading density from: {density_loc} | num_contexts: {num_contexts}, func_ind: {func_ind}")
    LOG.info(f"Writing animations to: {anim_dir}")
    outname = f"density_animate_c{num_contexts}_f{func_ind}_{frame_data_method}.mp4"
    anim_loc = anim_dir / outname
    if anim_loc.exists():
        LOG.warning(f"Animation already exists: {anim_loc}")
        anim_loc = anim_loc.parent / f"{anim_loc.name}.{int(time())}.mp4"
        LOG.warning(f'Writing animation to "{anim_loc}" instead')
    eval_points = np.arange(config['start'], config['end'], config['step'])
    my_anim = MyAnimator(
        density_loc,
        num_contexts=num_contexts,
        func_ind=func_ind,
        eval_points=eval_points,
    )
    my_anim.set_likelihoods()
    my_anim.set_frame_data(method=frame_data_method, frame_data=frame_data)
    my_anim.set_densities()

    vmax = np.nanquantile(my_anim.densities, max_quantile)
    vmin = np.nanquantile(my_anim.densities, min_quantile)
    norm = Normalize(vmin=vmin, vmax=vmax)
    if "norm" in config:
        if config["norm"] == "log":
            norm = LogNorm()
            LOG.warning("Using log norm, ignoring vmin and vmax")

    my_anim.set_contour_prefs(nlevels_min=nlevels_min, norm=norm)
    my_anim.set_first_frame()
    if only_first_frame:
        plt.show()
        return
    else:
        my_anim.write_animation(str(anim_loc), fps)


def make_heatmap(grd, vmin="vanilla", **kwargs):
    vanilla_ll = np.nanmax(grd[:, 0])
    if vmin == "vanilla":
        vmin = vanilla_ll
    ind = np.unravel_index(np.nanargmax(grd), grd.shape)
    indf = np.flip(ind)

    grd[grd == -np.inf] = np.nan

    ax = sns.heatmap(grd, vmin=vmin, **kwargs)
    plt.xlabel("trajectory length")
    plt.ylabel("number of trajectories")
    fig = plt.gcf()
    for tl, best_nt in enumerate(np.nanargmax(grd, axis=0)):
        col_max_ind = [tl, best_nt]
        rec = Rectangle((col_max_ind), 1, 1, fill=False, edgecolor="yellow", lw=2, clip_on=False)
        ax.add_patch(rec)
    best_rec = Rectangle((indf), 1, 1, fill=False, edgecolor="cyan", lw=2, clip_on=False)
    ax.add_patch(best_rec)
    return fig


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Create animations from density')
    parser.add_argument('i', help='input config file', type=Path)
    parser.add_argument('m', help='input models dir', type=Path)
    parser.add_argument('--only_first_frame', help='only show first frame', action='store_true')
    parser.set_defaults(only_first_frame=False)
    args = parser.parse_args()
    main(args.i, args.m, args.only_first_frame)
