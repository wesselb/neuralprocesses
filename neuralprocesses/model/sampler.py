import argparse
import matplotlib.pyplot as plt
import seaborn as sns
import concurrent.futures
import logging
import os
from collections import namedtuple
from functools import cached_property
from pathlib import Path

import h5py
import lab as B
import numpy as np
import torch
import yaml
from matplotlib.colors import LogNorm, Normalize, SymLogNorm
from scipy.stats import norm
from tqdm import tqdm

import experiment as exp
from neuralprocesses import torch as nps
from neuralprocesses.dist import UniformDiscrete, UniformContinuous
from neuralprocesses.model.trajectory import construct_trajectory_gens

LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def read_hdf5(hdf5_loc: Path):
    if not hdf5_loc.exists():
        raise FileNotFoundError(f"{hdf5_loc} does not exist")
    ss = SampleSet(hdf5_loc=hdf5_loc)
    return ss


class OuterSampler:
    def __init__(self, hdf5_dir, model, gen, traj_gen, device="cpu"):
        self.hdf_dir = hdf5_dir
        self.model = model
        self.gen = gen
        self.traj_gen = traj_gen
        self.tqdm = tqdm
        # these gens need to be retrieved somehow when loading from hdf5
        self.samples_sets = []
        # TODO: batch won't be defined at this level, will depend on the SampleSet
        self.density_loc = hdf5_dir / "density_grid.hdf5"
        with h5py.File(self.density_loc, "w") as f:
            md_grp = f.create_group("marginal_densities")
            tj_grp = f.create_group("trajectories")

        # ^ better to use a setter probably
        LOG.info(f'Making directory "{hdf5_dir.absolute()}"')
        hdf5_dir.mkdir(exist_ok=True)

        # these should be immutable properties
        self.trajectory_length = self.traj_gen.trajectory_length
        self.num_functions = None
        # TODO: num context won't be defined at this level, will depend on the SampleSet
        self.num_trajectories = None
        self.num_targets = None
        self.num_density_eval_locations = None

        self.device = device
        B.set_global_device(device)

    def generate_batch(self, batch_size=16, num_context=None):
        # This is messy since is changes the attributes of the generator.
        # Maybe fix at some point.
        self.gen.batch_size = batch_size
        if num_context is not None:
            # TODO: could also allow for ranges here?
            self.gen.num_context = UniformDiscrete(num_context, num_context)
        batch = self.gen.generate_batch()
        # choose batch with defined context? or use defined x_target?
        return batch

    def make_sample_sets(self, num_functions_per_context_size=10, num_samples=100, overwrite=False, context_range=(0, 10)):
        csizes = np.arange(context_range[0], context_range[1] + 1)
        # num_funcs_per_csize = num_functions / csizes.shape[0]
        num_functions = num_functions_per_context_size * csizes.shape[0]
        self.num_functions = num_functions
        self.num_targets = self.gen.num_target.upper #assumes upper and lower are the same
        # in other words, only one number of targets
        xt = torch.zeros((num_functions, 1, self.num_targets))
        yt = torch.zeros((num_functions, 1, self.num_targets))
        for ci, csize in tqdm(enumerate(csizes), total=csizes.shape[0], leave=False):
            start_ind = ci * num_functions_per_context_size
            end_ind = (ci + 1) * num_functions_per_context_size
            batch = self.generate_batch(batch_size=num_functions_per_context_size)
            xt[start_ind:end_ind, :, :] = batch["xt"]
            yt[start_ind:end_ind, :, :] = batch["yt"]
            contexts = batch["contexts"]
            # clear away existing sample sets
            self.samples_sets = []
            for i in self.tqdm(range(num_functions_per_context_size)):
                # hdf5_loc = self.hdf_dir / f"sample_set_{csize}|{i}.hdf5"
                hdf5_loc = self.density_loc
                # if hdf5_loc.exists() and not overwrite:
                if False:
                    ss = read_hdf5(hdf5_loc)
                    ss.gen = self.traj_gen
                else:
                    contexts_i = self.get_i_context(contexts, i)
                    ss = SampleSet(
                        hdf5_loc=hdf5_loc,
                        contexts=contexts_i,
                        gen=self.traj_gen,
                        overwrite=overwrite,
                        group_name=f"{csize}|{i}",
                    )
                    ss.tqdm = self.tqdm
                    # LOG.info("Creating sample set")
                    ss.create_samples(self.model, num_samples)
                self.samples_sets.append(ss)
            # TODO: won't need to have list because it will be a part of the file
            # or make the iterator a property which is defined by the file contents
            # use hdf5 external link
            # Will still have old sample sets lying around
            # Should just write everything to one hdf5
        # self.num_functions = num_functions_per_context_size
        self.num_trajectories = num_samples
        self.xt = xt
        self.yt = yt
        self.num_targets = self.yt.shape[2]
        # ^ above should be properties which are immutable and get their values
        # from hdf5
        # TODO: separate batches for each sample set to see impact of context size
        # Think about how to adapt this KL using ABC rejection sampling
        # Think about how to adapt to using chain rule of prob on joint
        return self.xt, self.yt

    @staticmethod
    def get_i_context(contexts, i):
        xc = contexts[0][0][i, :, :].reshape(1, 1, -1)
        yc = contexts[0][1][i, :, :].reshape(1, 1, -1)
        contexts_i = [(xc, yc)]
        return contexts_i

    def create_density_grid(
        self,
        density_eval="generated",
        density_kwargs=None,
        overwrite=False,
        workers=1,
    ):
        if len(self.samples_sets) == 0:
            raise ValueError("No sample sets created yet!")
        if self.density_loc.exists() and not overwrite:
            raise ValueError(f"{self.density_loc} already exists")
        with h5py.File(self.density_loc, "a") as f:
            md_grp = f["marginal_densities"]
            tj_grp = f["trajectories"]
            # md_grp = f.create_group("marginal_densities")
            # tj_grp = f.create_group("trajectories")
            for func_ind, ss in enumerate(self.samples_sets):
                md_grp.create_group(str(func_ind))
        LOG.info(f"Creating density grid for each {self.num_functions} sampled functions.")
        pbar = self.tqdm(enumerate(self.samples_sets), total=len(self.samples_sets))
        for func_ind, ss in pbar:
            self.inner_create_density_grid(
                ss,
                func_ind,
                density_eval=density_eval,
                density_kwargs=density_kwargs,
                workers=workers,
            )

    def get_targets_and_density_eval_points(
        self, func_ind, density_eval, density_kwargs
    ):
        targets = self.xt[func_ind, :, :].reshape(-1, 1)
        if density_eval == "generated":
            yt0 = self.yt[func_ind, :, :]
            y_targets = yt0.reshape(-1, 1)
        elif density_eval == "grid":
            # TODO: add option which gets these values automatically by looking at some
            # aspect of underlying generator.
            yt0 = torch.arange(
                density_kwargs["start"], density_kwargs["end"], density_kwargs["step"]
            )
            y_targets = yt0.repeat(targets.shape[0], 1)
        else:
            raise ValueError(f"density_eval must be one of 'generated' or 'grid'")
        self.num_density_eval_locations = y_targets.shape[1]
        return targets, y_targets

    def grid_loglikelihoods(self):
        grid = np.zeros((self.num_trajectories + 1, self.trajectory_length + 1))
        for nt in self.tqdm(range(self.num_trajectories + 1)):
            for tl in range(self.trajectory_length + 1):
                grid[nt, tl] = self.calc_loglikelihood(nt, tl)
        return grid

    def calc_loglikelihood(self, num_trajectories, trajectory_length):
        if num_trajectories > self.num_trajectories:
            raise ValueError(
                f"num_trajectories must be less than or equal to {self.num_trajectories}"
            )
        if trajectory_length > self.trajectory_length:
            raise ValueError(
                f"trajectory_length must be less than or equal to {self.trajectory_length}"
            )
        # TODO: use size from tensor here instead of the class attribute
        func_loglikelihoods = np.zeros(
            (self.num_functions, self.num_density_eval_locations)
        )
        for func_ind, _ in enumerate(self.samples_sets):
            with h5py.File(self.density_loc, "r") as f:
                grp = f["marginal_densities"][str(func_ind)]
                # grp = f[str(func_ind)]
                lh = grp["likelihoods"]
                # get mean likelihood of GMM components for all target points
                mn = lh[:, :num_trajectories, :, trajectory_length].mean(axis=1)
                # Get the log likelihood for each target point (under the GMM)
                target_lls = np.log(mn)
                # Sum the log likelihoods for each target point (not using chain rule to factorize, just summing)
                all_targets_ll = target_lls.sum()
                func_loglikelihoods[func_ind] = all_targets_ll
        return func_loglikelihoods.mean()

    def inner_create_density_grid(
        self,
        ss,
        func_ind,
        workers=1,
        density_eval="generated",
        density_kwargs=None,
    ):
        xc_all, yc_all = append_contexts_to_samples(ss.contexts, ss.traj)
        targets, y_targets = self.get_targets_and_density_eval_points(
            func_ind, density_eval, density_kwargs
        )
        with h5py.File(self.density_loc, "a") as f:
            grp = f["marginal_densities"][str(func_ind)]
            # Always the same number of targets throughout the batch.
            # Technically could add function draw as a dimension and add to tensor.
            grp.create_dataset(
                "likelihoods",
                (
                    self.num_targets,
                    self.num_trajectories,
                    self.num_density_eval_locations,
                    self.trajectory_length + 1,  # include trajectory length of 0.
                ),
            )
            # append to existing if adding more points
            # make something to store index which tells which experiment it is from
            # write all config yaml as attributes?
            # write github hash repo as attribute?
            grp.create_dataset("y_density_evaluation_points", data=y_targets.cpu())
            grp.create_dataset("x_targets", data=targets.cpu())  # (x_target locations)
            pbar = self.tqdm(
                range(self.trajectory_length + 1),
                total=self.trajectory_length + 1,
                leave=False,
            )
            for tl_ind in pbar:
                pbar.set_description(f"Trajectory length: {tl_ind}")
                # i=0 -> no AR (b/c no trajectory, only context)
                trunc_length = tl_ind + ss.num_contexts
                xc, yc = truncate_samples(xc_all, yc_all, trunc_length)

                xt = targets.reshape(-1, 1, 1)
                xt_ag = nps.AggregateInput((xt, 0))
                pred = self.model(xc, yc, xt_ag)
                # TODO: something different for many density eval points?
                densities = B.exp(pred.logpdf(y_targets))
                lls = B.transpose(densities).reshape(self.num_targets, self.num_trajectories, 1)
                llnp = lls.cpu().detach().numpy()
                grp["likelihoods"][:, :, :, tl_ind] = llnp


class SampleSet:
    def __init__(self, hdf5_loc: Path, contexts=None, gen=None, overwrite=False, group_name=None):
        self.hdf5_loc = hdf5_loc
        self.data = None
        self.gen = gen
        self.tqdm = tqdm
        self.dtype = torch.float32
        self.group_name = group_name

        if self.hdf5_loc.exists() and overwrite is False:
            if contexts is not None:
                LOG.warning(
                    f"HDF5 file already exists. Using existing file at {self.hdf5_loc}."
                )
                LOG.warning("ar_inputs, and contexts will be ignored.")
            else:
                LOG.warning(f"Loading from {self.hdf5_loc}")
        else:
            cx_np = contexts[0][0].cpu().detach().numpy().reshape(-1)
            cy_np = contexts[0][1].cpu().detach().numpy().reshape(-1)
            if group_name is not None:
                mode = "a"
            else:
                mode = "w"
            with h5py.File(self.hdf5_loc, mode) as f:
                grp = self.get_group(f)
                grp.attrs["trajectory_generator"] = str(gen)
                grp.attrs["sample_size"] = gen.trajectory_length
                grp.create_dataset("cx", data=cx_np)
                grp.create_dataset("cy", data=cy_np)

    def get_group(self, f):
        if self.group_name is not None:
            tj_grp = f["trajectories"]
            if self.group_name in tj_grp.keys():
                grp = tj_grp[self.group_name]
            else:
                grp = tj_grp.create_group(self.group_name)
        else:
            grp = f
        return grp

    @cached_property
    def trajectory_generator(self):
        with h5py.File(self.hdf5_loc, "r") as f:
            grp = self.get_group(f)
            traj_gen = grp.attrs["trajectory_generator"]
        return traj_gen

    @cached_property
    def sample_size(self):
        with h5py.File(self.hdf5_loc, "r") as f:
            grp = self.get_group(f)
            sample_size = int(grp.attrs["sample_size"])
        return sample_size

    @cached_property
    def contexts(self):
        with h5py.File(self.hdf5_loc, "r") as f:
            grp = self.get_group(f)
            cx = grp["cx"][:].reshape(1, 1, -1)  # not sure if -1 should be for last index
            cy = grp["cy"][:].reshape(1, 1, -1)
        cxt = B.to_active_device(B.cast(self.dtype, cx))
        cyt = B.to_active_device(B.cast(self.dtype, cy))
        fc = [(cxt, cyt)]
        return fc

    @cached_property
    def num_contexts(self):
        return len(self.contexts[0][0])

    def check_traj(self):
        with h5py.File(self.hdf5_loc, "r") as f:
            grp = self.get_group(f)
            if "y_traj" not in grp:
                raise Exception("Samples not yet created.")

    def get_traj(self, dset_name):
        self.check_traj()
        with h5py.File(self.hdf5_loc, "r") as f:
            grp = self.get_group(f)
            np_dim_traj = grp[dset_name][:]
            dim_traj = torch.Tensor(np_dim_traj).reshape(-1, 1, 1, self.sample_size)
        dim_traj = B.to_active_device(B.cast(self.dtype, dim_traj))
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
            grp = self.get_group(f)
            if "y_traj" in grp:
                LOG.warning("Samples already created. Skipping.")
                return

        # LOG.info(f"Initial context: {self.contexts}")
        # LOG.info(f"Generating {n_samples} trajectories of length {self.sample_size}")

        with h5py.File(self.hdf5_loc, "a") as f:
            grp = self.get_group(f)
            x_traj = grp.create_dataset(
                "x_traj", (n_samples, self.sample_size), dtype="float32"
            )
            y_traj = grp.create_dataset(
                "y_traj", (n_samples, self.sample_size), dtype="float32"
            )

            inner_samples = (
                1  # TODO: could tweak this for more speed, but less stochasticity
            )
            for i in self.tqdm(range(n_samples), leave=False):
                # TODO: assess whether this is fast enough, can I do it in parallel somehow?
                x_traj0 = self.gen.generate(self.contexts[0][0].cpu())
                x_traj0 = B.to_active_device(B.cast(self.dtype, x_traj0))
                y_traj0 = get_trajectories(model, x_traj0, inner_samples, self.contexts)
                x_traj[i] = x_traj0.cpu().detach().numpy()
                y_traj[i] = y_traj0.cpu().detach().numpy()


def get_trajectories(model, xi, n_mixtures, contexts):
    # Generate AR trajectories from the fixed selections in the domain.
    ag = nps.AggregateInput((xi, 0))
    mean, var, noiseless_samples, noisy_samples = nps.ar_predict(
        model, contexts, ag, num_samples=n_mixtures, order="given"
    )
    y_traj = noisy_samples.elements[0]
    return y_traj


def load_model(weights, name, device="cpu"):
    print("Name:", name)
    if name in ["sawtooth", "synthetic-audio", "simple-mixture"]:
        model = get_model_gen()
    elif name in ["real-audio"]:
        model = get_model_real_audio()
    else:
        raise ValueError(f"Unknown model name: {name}")
    model.load_state_dict(torch.load(weights, map_location=device)["weights"])
    return model


def get_generator(generator_kwargs, num_context=None, specific_x=None, device="cpu"):
    ARGS = namedtuple("args", generator_kwargs)
    args = ARGS(**generator_kwargs)
    config = {}

    gen_train, gen_cv, gens_eval = exp.data[args.data]["setup"](
        args,
        config,
        num_tasks_train=2**6,
        num_tasks_cv=2**6,
        num_tasks_eval=2**6,
        device=device,
    )
    if isinstance(num_context, int):
        num_context = UniformDiscrete(num_context, num_context)
    elif num_context is None:
        num_context = UniformDiscrete(0, 50)
    # TODO: Should probably get generator more manually...
    # That way have more control of what is passed.
    # including control of batch size
    # also just grabbing first generator from eval lack clarity.
    gen = gens_eval()[0][1]
    gen.num_context = num_context
    if specific_x is not None:
        if num_context.upper == 1:
            gen.dist_x_context = UniformContinuous(specific_x, specific_x)
        else:
            raise ValueError("Can't use specific x with many contexts")
    return gen


def get_model_gen():
    # Construct the model and load the weights.
    model = nps.construct_convgnp(  # where do I get these values? make sure they are correct
        dim_x=1,
        dim_y=1,
        unet_channels=(64,) * 6,
        points_per_unit=64,
        likelihood="het",
    )
    # model.load_state_dict(torch.load(weights, map_location="cpu")["weights"])
    return model


def get_model_real_audio():
    # Take these values from experiment/data/phone.py
    # TODO: This should be done in a more clear and automatable way
    num_channels = 8
    model = nps.construct_convgnp(
        points_per_unit=2,
        dim_x=1,
        dim_yc=(1,) * 1,
        dim_yt=1,
        likelihood="het",
        # conv_arch=args.arch,
        unet_channels=(128,) * num_channels,
        unet_strides=(1,) + (2,) * (num_channels - 1),
        conv_channels=64,
        conv_layers=6,
        conv_receptive_field=50,
        margin=0.25,  # 07_10
        # margin=1,  # 07_8
        encoder_scales=None,
        transform=None,
    )
    return model


def get_workers(workers):
    if workers == -1:
        return os.cpu_count()
    elif workers == "half":
        return int(os.cpu_count() / 2)
    elif workers is None:
        return 1
    else:
        return workers


def truncate_samples(xc_all, yc_all, max_len=None):
    # max_len here so that you can use a length less than the length of trajectory
    # used for the samples. Can be used to assess the impact of using different lengths
    # of trajectories.
    if max_len is not None:
        # LOG.info(f"Limiting trajectory length to {max_len}")
        xc = xc_all[..., :max_len]
        yc = yc_all[..., :max_len]
    else:
        xc = xc_all
        yc = yc_all
    return xc, yc


def generate_marginal_densities(
    out_marginal_densities, ss, model, targets, dxi, max_len=None, workers=None
):
    xc, yc = append_contexts_to_samples(ss.contexts, ss.traj)
    # TODO: iterate to full traj length and perform calc for all?
    # TODO: but reduce the number of x^(T) (targets) and only get density (dxi) for the
    # real value y^(T) (which we need to pass in to this function).
    xc, yc = truncate_samples(xc, yc, max_len)

    workers = get_workers(workers)
    with h5py.File(out_marginal_densities, "w") as f:
        # TODO: write out a reference to SampleSet
        # TODO: write out trajectory length truncation
        # Shape becomes max trajectory length so we have density for each traj length
        f.create_dataset("ad", (len(targets), len(dxi)), maxshape=(None, None))
        f.create_dataset("dxi", data=dxi.cpu())  # (grid_densities_calculated_on)
        f.create_dataset("targets", data=targets.cpu())  # (x_target locations)

        LOG.info(
            "{N(mu, var) = p_model(y_target|ar_context_i, x_target) "
            f"for y_target in targets (n={len(targets)}), "
            f"ar_context_i in trajectories (n={ss.y_traj.shape[0]}"
        )
        LOG.info("Generating predictive densities GMMs for all targets.")
        LOG.info("[[Sum(N(x=x0|mu, var)) for mu, var in preds] for x0 in dxi]")
        LOG.info(f"Using {workers} workers")

        dm = DensityMaker(xc, yc, model, dxi)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as executor:
            pbar = tqdm(executor.map(dm.generate, targets), total=len(targets))
            for i, ad0 in enumerate(pbar):
                f["ad"][i, :] = ad0

    return out_marginal_densities


def quick_dense(model, xc, yc, xt, yt):
    # xt, yt = xt_yt
    xt_yts = zip(xt, yt)
    l = list(xt_yts)

    xt = xt.reshape(-1, 1, 1)
    yt = yt.repeat(1, 2)
    xt_ag = nps.AggregateInput((xt, 0))
    pred = model(xc, yc, xt_ag)
    density = B.exp(pred.logpdf(yt))

    xt0, yt0 = l[1]
    xt0 = xt0.reshape(1, 1, -1)
    xt_ag0 = nps.AggregateInput((xt0, 0))
    # TODO: Find out how to do this while setting the seed for reproducibility
    # Could do this in a batched fashion, I think
    pred0 = model(xc, yc, xt_ag0)

    m2 = pred0.mean.elements[0].cpu().detach().numpy().reshape(-1)
    v2 = pred0.var.elements[0].cpu().detach().numpy().reshape(-1)
    # TODO: out put each component of the density and return for use in db
    densities = torch.zeros(len(m2), len(yt0))
    for i, (m, v) in enumerate(zip(m2, v2)):
        d0 = norm.pdf(yt0.cpu(), loc=m, scale=np.sqrt(v))
        densities[i, :] = torch.Tensor(d0)

    return density


class DensityMaker:
    def __init__(self, xc, yc, model):
        self.xc = xc
        self.yc = yc
        self.model = model

    def generate(self, xt_yt):
        xt, yt = xt_yt
        xt = xt.reshape(1, 1, -1)
        xt_ag = nps.AggregateInput((xt, 0))
        # TODO: Find out how to do this while setting the seed for reproducibility
        # Could do this in a batched fashion, I think
        pred = self.model(self.xc, self.yc, xt_ag)
        # TODO: for eval held out-loglik, only dxi of interest is the true value from
        # the observations.
        ad0 = plot_densities(pred, yt)
        return ad0


def append_contexts_to_samples(contexts, traj):
    # TODO: adapt to work with multiple outputs
    xc = traj[0]
    yc = traj[1]
    num_samples = traj[0].shape[0]
    repcx = contexts[0][0].repeat(num_samples, 1, 1, 1)
    repcy = contexts[0][1].repeat(num_samples, 1, 1, 1)
    xc = torch.cat((repcx, xc), axis=-1)
    yc = torch.cat((repcy, yc), axis=-1)

    return xc, yc


def plot_densities(pred, dxi):
    m2 = pred.mean.elements[0].cpu().detach().numpy().reshape(-1)
    v2 = pred.var.elements[0].cpu().detach().numpy().reshape(-1)

    # TODO: out put each component of the density and return for use in db
    densities = torch.zeros(len(m2), len(dxi))
    for i, (m, v) in enumerate(zip(m2, v2)):
        d0 = norm.pdf(dxi.cpu(), loc=m, scale=np.sqrt(v))
        densities[i, :] = torch.Tensor(d0)
    return densities


def get_dxi_and_targets(config):
    dr = config["density"]["range"]
    tr = config["targets"]["range"]
    dxi = torch.arange(dr["start"], dr["end"], dr["step"])
    targets = torch.arange(tr["start"], tr["end"], tr["step"])
    return dxi, targets


def get_device(device=None, gpu=None):
    if device:
        device = device
    elif gpu is not None:
        device = f"cuda:{gpu}"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    return device


def clean_config(config: dict) -> dict:
    # TODO: use defaultdict
    if "max_len" not in config:
        config["max_len"] = None
    if "num_samples" not in config:
        config["num_samples"] = None
    if "generator_kwargs" not in config:
        config["generator_kwargs"] = {}
    if "num_context" not in config:
        config["num_context"] = None
    if "specific_x" not in config:
        config["specific_x"] = None
    config["model_weights"] = Path(config["model_weights"])
    return config


def make_heatmap(grd, config, outdir):
    grd[grd == -np.inf] = np.nan
    ax = sns.heatmap(grd)
    plt.xlabel("trajectory length")
    plt.ylabel("number of trajectories")
    gname = config['trajectory']['generator']
    plt.title(f"{config['name']} log-likelihoods with {gname} trajectory.")
    plt.savefig(outdir / f"heatmap.png")
    plt.clf()
    # ax = sns.heatmap(grd, norm=SymLogNorm())
    # plt.xlabel("trajectory length")
    # plt.ylabel("number of trajectories")
    # gname = config['trajectory']['generator']
    # plt.title(f"{config['name']} log-likelihoods with {gname} trajectory.")
    # plt.savefig(outdir / f"heatmap_log.png")
    return outdir


def main(
    in_config: Path,
    out_sampler_dir: Path,
    device=None,
    gpu=None,
    workers=1,
    exist_ok=False,
    overwrite_ss=False,
):
    with open(in_config, "r") as f0:
        config = clean_config(yaml.safe_load(f0))

    device = get_device(device, gpu)
    B.set_global_device(device)

    data_generator = get_generator(
        config["generator_kwargs"],
        num_context=config["num_context"],
        specific_x=config["specific_x"],
        # TODO: make different context set sizes for different sample sets
        device=device,
    )
    trajectory_generator = construct_trajectory_gens(
        trajectory_length=config["trajectory"]["length"],
        x_range=(config["trajectory"]["low"], config["trajectory"]["high"]),
    )[config["trajectory"]["generator"]]
    model = load_model(config["model_weights"], config["name"], device=device)
    model = model.to(device)

    out_sampler_dir.mkdir(exist_ok=exist_ok)
    s = OuterSampler(
        out_sampler_dir, model, data_generator, trajectory_generator, device
    )
    LOG.info("Making Trajectories")
    s.make_sample_sets(
        num_functions_per_context_size=config["num_functions"],
        num_samples=config["n_mixtures"],
        overwrite=overwrite_ss,
        context_range=config["context_range"],
    )
    LOG.info("Getting all loglikelihoods")
    s.create_density_grid(density_eval="generated", overwrite=True, workers=workers)
    # TODO: overwrite not necessarily true
    grd = s.grid_loglikelihoods()
    make_heatmap(grd, config, out_sampler_dir)
    np.save(str(out_sampler_dir / "loglikelihoods_grid.npy"), grd)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("--in_config", help="input yaml config file", type=Path)
    parser.add_argument("--out_sampler_dir", help="output directory", type=Path)
    parser.add_argument("--device", help="device to use", default=None)
    parser.add_argument("--gpu", help="gpu to use", type=int, default=None)
    parser.add_argument(
        "--workers", help="number of workers to use", type=int, default=None
    )
    parser.add_argument("--exist_ok", dest="exist_ok", action="store_true")
    parser.add_argument("--no-exist_ok", dest="exist_ok", action="store_false")
    parser.add_argument("--overwrite_ss", dest="overwrite_ss", action="store_true")
    parser.add_argument("--no-overwrite_ss", dest="overwrite_ss", action="store_false")

    parser.set_defaults(exist_ok=True)
    parser.set_defaults(overwrite_ss=False)

    args = parser.parse_args()
    main(
        args.in_config,
        args.out_sampler_dir,
        args.device,
        args.gpu,
        args.workers,
        args.exist_ok,
        args.overwrite_ss,
    )
