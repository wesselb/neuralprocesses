import logging
from datetime import datetime
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
from time import time
import seaborn as sns
import yaml
from matplotlib.patches import Rectangle
import matplotlib.animation as animation

import scripts.ar_marginal as sam
from neuralprocesses.ar.sampler import (
    Groups,
    Datasets,
    get_func_expected_ll,
)
import matplotlib as mpl

mpl.rcParams["figure.dpi"] = 300
plt.rcParams.update({"figure.figsize": (14, 6)})


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


class MyAnimator:
    def __init__(self, density_loc, num_contexts, func_ind):
        self.density_loc = density_loc
        self.num_contexts = num_contexts
        self.func_ind = func_ind

        self.lh = None
        self.xt = None
        self.eval_points = None
        self.true_y_targets = None
        self.cx = None
        self.cy = None
        self.total_num_trajectories = None
        self.total_trajectory_len = None

        self.frame_data = None

        self.densities = None
        self.targets = None
        self.density_eval_locations = None

        self.vmax = None
        self.nlevels = None

        self.figure = None
        self.ax = None
        self.cont = None

    def set_densities(self, nlevels_min=100, quantile=0.95):
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
        self.densities = np.stack(densities)
        self.vmax = np.nanquantile(self.densities, quantile)
        self.nlevels = min(nlevels_min, int(density_eval_locations.shape[0] / 2))
        self.targets = targets  # these should be the same each time
        self.density_eval_locations = density_eval_locations
        LOG.info(f"Using {self.nlevels} levels with vmax={self.vmax:.2f}")
        LOG.info(f"Density shape: {self.densities.shape}")

    def set_frame_data(self, method="all_trajectory_lengths", frame_data=None):
        # First column is number of trajectories, second column is trajectory length
        if method == "all_trajectory_lengths":
            frame_data = np.empty((self.total_trajectory_len, 2), dtype=int)
            frame_data[:, 0] = self.total_num_trajectories
            frame_data[:, 1] = np.arange(self.total_trajectory_len)
        elif method == "all_num_trajectories":
            frame_data = np.empty((self.total_num_trajectories - 1, 2))
            frame_data[:, 0] = np.arange(self.total_num_trajectories)[1:]  # skip first bc nan
            frame_data[:, 1] = self.total_trajectory_len - 1
        elif method == "provided":
            if frame_data is not None:
                frame_data = np.array(frame_data) # coerce to numpy if list
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

            lh = grp[Datasets.LIKELIHOODS.value][:]
            xt = grp["x_targets"][:]
            eval_points = grp["y_density_evaluation_points"][:]
            cx = t0["cx"][self.func_ind, :]
            cy = t0["cy"][self.func_ind, :]

        self.lh = lh
        self.xt = xt
        self.eval_points = eval_points
        self.true_y_targets = eval_points[:, 0]
        self.cx = cx
        self.cy = cy
        self.total_num_trajectories = lh.shape[1]
        self.total_trajectory_len = lh.shape[-1]

    def get_density_grid(self, num_trajectories, trajectory_length):
        density = self.lh[:, :num_trajectories, :, trajectory_length]
        gmm = density.mean(axis=1).T
        x_order = np.argsort(self.xt, axis=0).reshape(-1)
        y_order = np.argsort(-self.eval_points[0], axis=0).reshape(-1)
        od = gmm[:, x_order][y_order, :]

        targets = self.xt[x_order].reshape(-1)
        density_eval_locations = self.eval_points[0][y_order].reshape(-1)
        return od, targets, density_eval_locations

    def set_first_frame(self):
        num_traj0, traj_len0 = self.frame_data[0]
        od = self.densities[0]
        targets = self.targets
        density_eval_locations = self.density_eval_locations
        LOG.info(f"Trajectory length: {traj_len0} | Number of trajectories: {num_traj0}")

        self.figure, self.ax = plt.subplots(figsize=(20, 5))
        self.ax.set_xlim(targets.min(), targets.max())
        self.ax.set_ylim(density_eval_locations.min(), density_eval_locations.max())

        x = targets
        y = density_eval_locations
        self.cont = plt.contourf(
            x, y, od, self.nlevels, vmax=self.vmax, zorder=1
        )  # first image on screen
        plt.scatter(
            self.cx, self.cy, s=150, marker="+", color="red", label="contexts", zorder=3
        )
        plt.scatter(
            self.xt, self.true_y_targets, color="black", label="targets", zorder=2
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
        LOG.info(f"Trajectory length: {traj_len0} | Number of trajectories: {num_traj0}")
        num_traj0, traj_len0 = self.frame_data[frame_index]
        if self.cont is None:
            raise Exception("First frame not set")
        for c in self.cont.collections:
            c.remove()  # removes only the contours, leaves the rest intact
        cont = plt.contourf(
            targets,
            density_eval_locations,
            od,
            self.nlevels,
            cmap="viridis",
            vmax=self.vmax,
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


def observe_density():
    density_dir = Path("../../../models/ar_test-sawtooth-ll_small_emanate")
    num_contexts = 1
    func_ind = 0

    # frame_data_method = "all_num_trajectories"
    # frame_data_method = "all_trajectory_lengths"
    frame_data_method = "provided"
    frame_data = [[1, 1], [2, 1], [4, 1], [8, 1], [16, 1], [32, 1], [64, 1], [128, 1], [256, 1], [512, 1], [1024, 1]]

    nlevels_min = 100
    quantile = 0.95
    fps = 8

    anim_dir = density_dir / "animations"
    anim_dir.mkdir(exist_ok=True)
    density_loc = density_dir / "densities.hdf5"

    LOG.info(f"Loading density from: {density_loc} | num_contexts: {num_contexts}, func_ind: {func_ind}")
    LOG.info(f"Writing animations to: {anim_dir}")
    anim_loc = anim_dir / f"density_animate_c{num_contexts}_f{func_ind}_{frame_data_method}.mp4"
    if anim_loc.exists():
        LOG.warning(f"Animation already exists: {anim_loc}")
        anim_loc = anim_loc.parent / f"{anim_loc.name}.{int(time())}.mp4"
        LOG.warning(f"Writing animation to \"{anim_loc}\" instead")
        # raise Exception(f"Animation already exists: {anim_loc}")

    my_anim = MyAnimator(density_loc, num_contexts, func_ind)
    my_anim.set_likelihoods()
    my_anim.set_frame_data(method=frame_data_method, frame_data=frame_data)
    my_anim.set_densities(nlevels_min=nlevels_min, quantile=quantile)
    my_anim.set_first_frame()
    my_anim.write_animation(str(anim_loc), fps)


def main():
    tname = "test-sawtooth-ll_small_emanate"
    with open(f"../config/{tname}.yaml") as f0:
        config = sam.clean_config(yaml.safe_load(f0))

    all_grd = np.load(
        "../../../models/ar_test-sawtooth-ll_small_emanate/loglikelihoods_grid.npy")
    all_grd[all_grd == -np.inf] = np.nan
    grd = all_grd[:, :]
    ind = np.unravel_index(np.nanargmax(grd), grd.shape)
    indf = np.flip(ind)
    all_grd
    np.nanmax(grd)
    vmin = np.nanquantile(grd, 0.05)
    out_of_bounds_inds = np.transpose((grd < vmin).nonzero())
    vanilla_ll = np.nanmax(grd[:, 0])
    print(vanilla_ll)
    best_ll = grd[ind]
    print(best_ll)
    print(best_ll - vanilla_ll)
    worse_than_baseline_inds = np.transpose((grd < vanilla_ll).nonzero())
    # norm=None
    # norm = SymLogNorm(linthresh=0.01)
    # ax = sns.heatmap(grd, norm=norm, vmax=-20)
    ax = sns.heatmap(grd, vmin=vanilla_ll)
    plt.xlabel("trajectory length")
    plt.ylabel("number of trajectories")
    plt.title(
        f"{config['name']} log-likelihoods with {config['trajectory']['generator']} trajectory."
    )
    todays_date = f"{datetime.now():%Y-%m-%d}"
    outfile = f"../reports/figures/{tname}_{todays_date}.png"
    print(outfile)
    plt.savefig(outfile)
    np.save(f"../reports/figures/{tname}_{todays_date}.npy", grd)
    for tl, best_nt in enumerate(np.nanargmax(grd, axis=0)):
        col_max_ind = [tl, best_nt]
        ax.add_patch(
            Rectangle(
                (col_max_ind), 1, 1, fill=False, edgecolor="yellow", lw=2, clip_on=False
            )
        )
    # for i, j in out_of_bounds_inds:
    #     ax.add_patch(Rectangle(((j, i)), 1, 1, fill=False, edgecolor='green', lw=2, clip_on=False))
    # for i, j in worse_than_baseline_inds:
    #     ax.add_patch(Rectangle(((j, i)), 1, 1, fill=False, edgecolor='blue', lw=1, clip_on=False))
    for i, j in worse_than_baseline_inds:
        ax.add_patch(
            Rectangle(((j, i)), 1, 1, fill=True, edgecolor="blue", lw=1, clip_on=False)
        )
    ax.add_patch(
        Rectangle((indf), 1, 1, fill=False, edgecolor="cyan", lw=2, clip_on=False)
    )
    plt.show()


if __name__ == "__main__":
    # parser = argparse.ArgumentParser(description='Say hello')
    # parser.add_argument('i', help='input txt file')
    # args = parser.parse_args()
    # main(args.i)
    # main()
    observe_density()
