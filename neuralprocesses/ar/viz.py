import logging
import click
from pathlib import Path
from tqdm import tqdm

import h5py
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm, Normalize
from time import time
import seaborn as sns
import yaml
from matplotlib.patches import Rectangle
import matplotlib.animation as animation
import matplotlib.transforms as mtransforms

from neuralprocesses.ar.loglik import get_func_expected_ll
from matplotlib.transforms import Affine2D
from matplotlib.collections import PathCollection
from neuralprocesses.ar.enums import Groups, Datasets
import neuralprocesses.ar.loglik as ll
import neuralprocesses.ar.utils as utils
from neuralprocesses.ar.baseline import FunctionInfo
import matplotlib as mpl
import pandas as pd

from neuralprocesses.ar.utils import get_function_density
import neuralprocesses.ar.baseline as bl

mpl.rcParams["figure.dpi"] = 300


logging.basicConfig(level=logging.INFO)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def get_exp_roll_diff(ll_df0):
    ll_df0 = ll_df0.copy()
    # Get rolling means within each function
    func_df = ll_df0.groupby(
        ["overall_func_ind", "trajectory_length", "num_trajectories"]).agg("mean")
    gb = func_df.groupby(["overall_func_ind", "trajectory_length"])

    roll_diff_df = gb.apply(shape_df).reset_index()
    # Get expectation over all functions at each num_traj
    rdf_mean = (
        roll_diff_df
            .groupby(["trajectory_length", "num_trajectories"])
            .agg({"log_likelihood": "mean"})
    )
    return rdf_mean


def get_conv_nt(df0, conv_threshold):
    """Get number of trajectories to converge"""
    df0 = df0.copy()
    csum = (df0["log_likelihood"] > conv_threshold).astype(int).cumsum()
    conv_nt = csum[csum == csum.max()].reset_index()['num_trajectories'].min()
    # TODO: right now this will say convervenge for the last number of trajectories if never stops increasing (i.e. doesnt converge yet)
    return conv_nt


def get_convergence(ll_df0, mn_baseline, conv_threshold, plot=False):
    rdf_mean = ll_df0.pipe(get_exp_roll_diff)
    gb = rdf_mean.groupby("trajectory_length")
    conv_nts = gb.apply(get_conv_nt, conv_threshold=conv_threshold)
    if plot is True:
        ll_fig = plot_expected_lls_by_traj(ll_df0, mn_baseline)
        fig = plot_rolling_differences(rdf_mean, conv_threshold)
        for cnt in conv_nts:
            vline = ll_fig.axes[0].axvline(x=cnt, color="yellow", linestyle="--")
        ll_fig.show()
    return conv_nts


def get_convergence2(ll_df0, mn_baseline, conv_threshold, plot=False, ax=None):
    # ll_df0 = ll_df0[~ll_df0["trajectory_length"] > 8]
    rdf_mean = ll_df0.pipe(get_exp_roll_diff)
    gb = rdf_mean.groupby("trajectory_length")
    conv_nts = gb.apply(get_conv_nt, conv_threshold=conv_threshold)
    if plot is True:
        ll_fig = plot_expected_lls_by_traj(ll_df0, mn_baseline, ax=ax)
        fig = plot_rolling_differences(rdf_mean, conv_threshold)

        ncolors = len(ll_df0.trajectory_length.unique()) - 1
        pal = sns.color_palette("flare", n_colors=ncolors)
        pal = ["blue"] + list(pal)

        if ax is None:
            ax0 = ll_fig.axes[0]
        else:
            ax0 = ax

        for cnt, color in zip(conv_nts, pal):
            vline = ax0.axvline(x=cnt, color=color, linestyle="--")
        # ll_fig.show()
    else:
        ll_fig = None
    return conv_nts, ll_fig


def plot_expected_lls_by_traj(ll_df0, mn_baseline, ax=None):
    func_df = ll_df0.groupby(["num_trajectories", "trajectory_length"]).agg("mean")
    top = func_df['log_likelihood'].max()
    fdf = func_df.reset_index()

    vanilla_val = fdf.loc[fdf['trajectory_length'] == 0, "log_likelihood"].max()
    ll_df0.loc[ll_df0["trajectory_length"] == 0, "trajectory_length"] = "Vanilla"

    ll_df0.trajectory_length.unique()
    ncolors = len(ll_df0.trajectory_length.unique()) - 1
    pal = sns.color_palette("flare", n_colors=ncolors)
    pal = ["blue"] + list(pal)

    lp = sns.lineplot(
        data=ll_df0,
        x="num_trajectories",
        y="log_likelihood",
        hue="trajectory_length",
        # style="context_size",
        ci=None,
        # legend="full",
        palette=pal,
        ax=ax,
    )
    # lp.set(yscale="log")
    # plt.gca().set_ylim(bottom=mn_baseline)
    plt.ylim(mn_baseline, top)
    # plt.ylim(vanilla_val, top)
    # plt.ylim(vanilla_val * 0.9, top * 1.1)
    fig = plt.gcf()
    plt.show()
    return fig


def plot_rolling_differences(rdf_mean, conv_threshold=0.0025):
    ax = sns.lineplot(data=rdf_mean, x="num_trajectories", y="log_likelihood",
                      hue="trajectory_length", ci=None)
    fig = ax.get_figure()
    ax.set_ylim(bottom=-0.01, top=0.05)
    ax.axhline(y=conv_threshold, color="red", linestyle="--")
    fig.show()
    return fig


def shape_df(df0):
    """Get diffs and apply rolling mean to convergence plot"""
    df0 = df0.copy()
    df1 = (
        df0['log_likelihood']
            .diff()
            .reset_index()
            .set_index("num_trajectories")[['log_likelihood']]
    )
    df1["log_likelihood"] =df1["log_likelihood"].rolling(5).mean()
    return df1


def get_perc_increases(df1, ndf, ddf):
    df1 = df1.copy()
    df1 = df1.set_index(["Context Size", "Function Index"])
    df1["log-likelihood"] = df1["log-likelihood"] - ndf["log-likelihood"]
    # Do I really need to do the mean thing here? I lose my bootstrap error bars
    tdf1 = df1.groupby(["Context Size"]).agg({"log-likelihood": "mean"})
    tdf1 = tdf1.reset_index()
    tdf1["method"] = 8 # <- this has no impact
    tdf1 = tdf1.set_index(["Context Size"])

    tddf = ddf.groupby(["Context Size", "method"]).agg({"log-likelihood": "mean"})
    tddf = tddf.reset_index()
    tddf = tddf.set_index(["Context Size"])

    ll_perc_increase = (tdf1["log-likelihood"] - tddf["log-likelihood"]) / tddf[
        "log-likelihood"]
    plot_df = ll_perc_increase.reset_index()
    return plot_df


def get_increase_over_vanilla(df1, vdf):
    df1 = df1.copy()
    df1 = df1.set_index(["Context Size", "Function Index"])
    df1["log-likelihood"] = df1["log-likelihood"] - vdf["log-likelihood"]
    plot_df = df1.reset_index()
    return plot_df


def get_scaled_lls(mdf):

    def scale(df0):
        cf_df = df0[df0['method'] != "Naive"].copy()
        llmin = cf_df['log-likelihood'].min()
        llmax = cf_df['log-likelihood'].max()
        scaled = (cf_df['log-likelihood'] - llmin) / (llmax - llmin)
        scaled = scaled * (1 - 0) + 0
        cf_df['log-likelihood'] = scaled
        return cf_df

    gb = mdf.groupby(["Context Size", "Function Index"])
    scaled_df = gb.apply(scale)
    return scaled_df


def get_melted_ll_df(density_loc, ll_npy):
    num_targets, context_sizes, num_func_per_context, cs_map, max_traj_len = bl.get_num_targets(density_loc)
    all_ll = np.load(ll_npy)

    lldfs = []
    i = 0
    for cs, func_inds in cs_map.items():
        for i in func_inds:
            llm = all_ll[:, :, i] / num_targets
            ll_df = (
                pd.DataFrame(llm)
                    .reset_index()
                    .rename(columns={"index": "num_trajectories"})
                    .melt(
                    id_vars=["num_trajectories"],
                    value_name="log_likelihood",
                    var_name="trajectory_length",
                )
                    .assign(overall_func_ind=i, context_size=cs)
            )
            lldfs.append(ll_df)
    if i != (all_ll.shape[-1] - 1):
        raise ValueError("Size is wrong!")
    ll_df = pd.concat(lldfs).reset_index(drop=True)
    return ll_df


def plot_naive_vanilla_ar_bars(density_loc, nt, tl):
    df = bl.get_func_ll_df(density_loc, tl=tl, nt=nt, normalize=True)
    mn_baseline = df.loc[df["method"] == "Naive", "log-likelihood"].mean()
    # ^ does not matter which AR params we choose
    # sns.barplot(x="Context Size", y="log-likelihood", hue="method", data=df)
    # plt.show()
    return mn_baseline


class DensityComparisonPlot:
    def __init__(
        self,
        fi,
        fig,
        eval_points,
        slice_inds,
        pal,
        xmin=None,
        xmax=None,
        ha_shift=0,
        marg_xmin=None,
        marg_xmax=None,
    ):
        self.fi = fi
        self.fig = fig
        self.eval_points = eval_points
        self.slice_inds = slice_inds
        self.pal = pal
        self.xmin = xmin
        self.xmax = xmax
        self.ha_shift = ha_shift
        self.marg_xmin = marg_xmin
        self.marg_xmax = marg_xmax

    def make_row_plots(self, density_ax, marginal_ax, nt, tl):
        density, targets, deval = density_contour(
            self.fi,
            density_ax,
            self.fig,
            self.eval_points,
            nt,
            tl,
            self.xmin,
            self.xmax,
            self.ha_shift,
        )
        ldfs = []
        for i, c in zip(self.slice_inds, self.pal):
            xt0 = targets[i]
            ldf = pd.DataFrame(
                {"eval_locations": self.eval_points, "density": density[:, i]}
            )
            ldf["eval_point"] = xt0
            ldfs.append(ldf)
            _ = density_ax.axvline(xt0, color=c, linestyle="--")
        ldf = pd.concat(ldfs).reset_index(drop=True)
        density_slice(ldf, marginal_ax, self.pal, self.marg_xmin, self.marg_xmax)

        for i, c in zip(self.slice_inds, self.pal):
            yt0 = self.fi.yt[i]
            marginal_ax.axhline(yt0, color=c, linestyle="--", alpha=0.5)


def density_slice(ldf, ax, palette, xmin=None, xmax=None, flip=True):
    if flip:
        lineplot = lineplot_plusplus
    else:
        lineplot = sns.lineplot
    lines_ax = lineplot(
        ax=ax,
        x="eval_locations",
        y="density",
        data=ldf,
        hue="eval_point",
        palette=palette,
        alpha=1,
        legend=False,
    )
    lines_ax.tick_params(
        axis="y", which="both", bottom=False, top=False, labelbottom=False
    )
    if xmin is not None and xmax is not None:
        lines_ax.set_xlim(xmin, xmax)
    lines_ax.set_ylabel("")
    lines_ax.set_xlabel("")


def density_contour(fi, ax, fig, eval_points, nt, tl, xmin=None, xmax=None, ha_shift=0):
    (_, _), (density, targets, deval) = fi.plot(
        num_trajectories=nt,
        trajectory_length=tl,
        eval_points=eval_points,
        figure=fig,
        ax=ax,
    )
    ll = fi.get_expected_loglik(nt, tl, normalize=True)
    if (xmin is not None) and (xmax is not None):
        ax.set_xlim(xmin, xmax)  # <- tune this

    l = rf"$R={tl} \mid \mathbb{{E}}[\mathcal{{L}}]={ll:.2f}$"
    trans = mtransforms.ScaledTranslation(10 / 72, -5 / 72, fig.dpi_scale_trans)
    ax.text(
        ha_shift,
        1.0,
        l,
        transform=ax.transAxes + trans,
        fontsize="small",
        verticalalignment="top",
        fontfamily="serif",
        bbox=dict(facecolor="0.7", edgecolor="none", pad=2.0),
    )
    return density, targets, deval


def lineplot_plusplus(orientation="horizontal", **kwargs):
    line = sns.lineplot(**kwargs)

    r = Affine2D().scale(sx=1, sy=-1).rotate_deg(90)
    for x in line.images + line.lines + line.collections:
        trans = x.get_transform()
        x.set_transform(r + trans)
        if isinstance(x, PathCollection):
            transoff = x.get_offset_transform()
            x._transOffset = r + transoff

    old = line.axis()
    line.axis(old[2:4] + old[0:2])
    xlabel = line.get_xlabel()
    line.set_xlabel(line.get_ylabel())
    line.set_ylabel(xlabel)

    return line


def get_function_predictive_params(density_loc, context_size, function_index):
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
        self.targets = targets  # should be the same each time
        self.densities = np.stack(densities)

    def set_contour_prefs(
        self, nlevels_min=100, max_quantile=0.95, min_quantile=0.05, norm=None
    ):
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
        self.eval_points = np.concatenate(
            [
                self.true_y_targets,
                self.eval_points,
                cy.reshape(-1),
            ]
        )

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
            xt_ordered, yt_ordered, color="black", label="targets", zorder=2, alpha=0.5
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
    in_config = utils.make_menu(active_config_dir, model_dir)
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
        cs = ac["num_contexts"]
        func_ind = ac["func_ind"]
        pbar.set_description(f"Context Size: {cs} | Function: {func_ind}")
        make_animation(exp_dir, ac, only_first_frame)
    LOG.info(f'Saved to "{exp_dir}".')


@click.group()
@click.option("--debug/--no-debug", default=False)
def cli(debug):
    click.echo("Debug mode is %s" % ("on" if debug else "off"))


@cli.command()
@click.argument("active-config-dir", type=Path)
@click.argument("model-dir", type=Path)
@click.option("--only_first_frame", default=False, type=bool)
def animate(active_config_dir: Path, model_dir: Path, only_first_frame: bool = False):
    """Create function animations as defined in config file."""
    config, exp_dir = setup(active_config_dir, model_dir)
    make_animations(config, exp_dir, only_first_frame)


@cli.command()
@click.argument("active-config-dir", type=Path)
@click.argument("model-dir", type=Path)
def heatmap(active_config_dir: Path, model_dir: Path):
    """Make heatmaps for each context size"""
    _, exp_dir = setup(active_config_dir, model_dir)
    make_heatmaps(exp_dir)


def set_mpl_params_latex():
    mpl.rcParams["figure.dpi"] = 300

    tex_fonts = {
        # Use LaTeX to write all text
        "text.usetex": True,
        "font.family": "serif",
        # Use 10pt font in plots, to match 10pt font in document
        "axes.labelsize": 10,
        "font.size": 10,
        # Make the legend/label fonts a little smaller
        "legend.fontsize": 8,
        "xtick.labelsize": 8,
        "ytick.labelsize": 8,
        'text.latex.preamble': r'\usepackage{amsfonts}',
    }
    mpl.rcParams.update(tex_fonts)


@cli.command()
@click.argument("active-config-dir", type=Path)
@click.argument("model-dir", type=Path)
def slice(active_config_dir: Path, model_dir: Path):
    """Make heatmaps for each context size"""
    config, exp_dir = setup(active_config_dir, model_dir)
    outer_make_slices(config, exp_dir)


def outer_make_slices(config, exp_dir):
    set_mpl_params_latex()
    pbar = tqdm(config["slices"])
    for ac in pbar:
        cs = ac["num_contexts"]
        func_ind = ac["func_ind"]
        pbar.set_description(f"Context Size: {cs} | Function: {func_ind}")
        make_slices(ac, exp_dir)
    LOG.info(f'Saved to "{exp_dir}".')


def make_slices(cfg, exp_dir):
    density_loc = exp_dir / "densities.hdf5"
    slice_dir = exp_dir / "slices"
    slice_dir.mkdir(exist_ok=True)

    nc = cfg["num_contexts"]
    func_ind = cfg["func_ind"]
    slice_inds = cfg["slice_inds"]

    fi = FunctionInfo.from_h5(
        density_loc,
        num_contexts=nc,
        func_ind=func_ind
    )

    ep_dct = cfg["eval_points"]
    eval_points = np.arange(ep_dct['start'], ep_dct['end'], ep_dct['step'])
    pal = sns.color_palette("bright", n_colors=len(slice_inds))

    fig_dim = utils.set_size(fraction=1, subplots=(2, 5), height_frac=2)
    fig, axd = plt.subplot_mosaic(
        [['upper left', 'upper left', 'upper left', 'upper left', 'upper right'],
         ['lower left', 'lower left', 'lower left', 'lower left', 'lower right']],
        figsize=fig_dim, constrained_layout=True
    )

    dcp = DensityComparisonPlot(
        fi, fig, eval_points, slice_inds, pal, xmin=cfg["xmin"],
        xmax=cfg["xmax"], ha_shift=cfg["ha_shift"], marg_xmin=cfg["marg_xmin"],
        marg_xmax=cfg["marg_xmax"]
    )
    tl0 = 0
    tl1 = 8
    dcp.make_row_plots(axd["upper left"], axd["upper right"], 128, tl0)
    dcp.make_row_plots(axd["lower left"], axd["lower right"], 128, tl1)
    axd["upper left"].legend(prop={'size': 7.1}, framealpha=0.3)
    title = fr"{cfg['title_name']} Marginal Density Change ($R={tl0} \rightarrow R={tl1}$)"
    fig.suptitle(title)

    fig.savefig(slice_dir / f"slices_{nc}_{func_ind}.png")


def make_heatmaps(exp_dir: Path):
    ll_loc = exp_dir / "log_likelihoods.npy"
    density_loc = exp_dir / "densities.hdf5"
    heatmap_dir = exp_dir / "heatmaps"
    heatmap_dir.mkdir(exist_ok=True)
    lc = ll.LikelihoodCalculator(density_loc)
    all_lls = np.load(str(ll_loc))
    plt.clf()

    ll_fig, conv_nts = barplot.get_single_conv_plot(exp_dir, mn_baseline, cs="all", conv_threshold=conv_threshold, ax=axd['left'])

    for cs, valid_funcs in lc.context_size_map.items():
        valid_funcs = lc.context_size_map[cs]
        valid_funcs = valid_funcs[:]
        cs_mn = all_lls[:, :, valid_funcs].mean(axis=-1)
        make_heatmap(cs_mn)
        plt.title(f"Expected Log-likelihood for Functions with Context Size {cs}")
        plt.savefig(heatmap_dir / f"{cs}.png")
        plt.clf()
    LOG.info(f'Written heatmaps to "{heatmap_dir}".')
    return heatmap_dir


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

    LOG.info(
        f"Loading density from: {density_loc} | num_contexts: {num_contexts}, func_ind: {func_ind}"
    )
    LOG.info(f"Writing animations to: {anim_dir}")
    outname = f"density_animate_c{num_contexts}_f{func_ind}_{frame_data_method}.mp4"
    anim_loc = anim_dir / outname
    if anim_loc.exists():
        LOG.warning(f"Animation already exists: {anim_loc}")
        anim_loc = anim_loc.parent / f"{anim_loc.name}.{int(time())}.mp4"
        LOG.warning(f'Writing animation to "{anim_loc}" instead')
    eval_points = np.arange(config["start"], config["end"], config["step"])
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


def make_heatmap(grd, vmin="vanilla", conv_nts=None, rotate=False, **kwargs):
    vanilla_ll = np.nanmax(grd[:, 0])
    if vmin == "vanilla":
        vmin = vanilla_ll
    ind = np.unravel_index(np.nanargmax(grd), grd.shape)

    if rotate is False:
        indf = np.flip(ind)
    else:
        indf = ind

    grd[grd == -np.inf] = np.nan

    if rotate is False:
        ax = sns.heatmap(grd, vmin=vmin, **kwargs)
    else:
        ax = sns.heatmap(grd.T, vmin=vmin, **kwargs)
    plt.xlabel("trajectory length")
    plt.ylabel("number of trajectories")
    fig = plt.gcf()

    for tl, best_nt in enumerate(np.nanargmax(grd, axis=0)):
        if rotate is False:
            col_max_ind = [tl, best_nt]
        else:
            col_max_ind = [best_nt, tl]
        rec = Rectangle(
            (col_max_ind), 1, 1, fill=False, edgecolor="yellow", lw=2, clip_on=False
        )
        ax.add_patch(rec)
    best_rec = Rectangle(
        (indf), 1, 1, fill=False, edgecolor="cyan", lw=2, clip_on=False
    )
    ax.add_patch(best_rec)

    if conv_nts is not None:
        for tl, conv in conv_nts.items():
            if rotate is False:
                col_max_ind = [tl, conv]
            else:
                col_max_ind = [conv, tl]
            rec = Rectangle(
                (col_max_ind), 1, 1, fill=False, edgecolor="green", lw=1, clip_on=False
            )
            ax.add_patch(rec)
    if rotate is True:
        ax.invert_yaxis()

    return fig


if __name__ == "__main__":
    cli()
