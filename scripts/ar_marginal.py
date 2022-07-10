import argparse
import concurrent.futures
import logging
import os
from pathlib import Path
from typing import List, Tuple

import h5py
import numpy as np
import torch
import yaml
from scipy.stats import norm
from tqdm import tqdm

import neuralprocesses.torch as nps
from neuralprocesses.model.sampler import SampleSet, read_hdf5
from neuralprocesses.model.trajectory import construct_trajectory_gens

LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def get_workers(workers):
    if workers == -1:
        return os.cpu_count()
    elif workers == "half":
        return int(os.cpu_count() / 2)
    elif workers is None:
        return 1
    else:
        return workers


def truncate_samples(xc, yc, max_len=None):
    # max_len here so that you can use a length less than the length of trajectory
    # used for the samples. Can be used to assess the impact of using different lengths
    # of trajectories.
    if max_len is not None:
        LOG.info(f"Limiting trajectory length to {max_len}")
        xc = xc[..., : max_len + 1]
        yc = yc[..., : max_len + 1]
    return xc, yc


def generate_marginal_densities(
    out_marginal_densities, ss, model, targets, dxi, max_len=None, workers=None
):
    xc, yc = append_contexts_to_samples(ss.contexts, ss.traj)
    xc, yc = truncate_samples(xc, yc, max_len)

    workers = get_workers(workers)
    with h5py.File(out_marginal_densities, "w") as f:
        # TODO: write out a reference to SampleSet
        # TODO: write out trajectory length truncation
        f.create_dataset("ad", (len(targets), len(dxi)), maxshape=(None, None))
        f.create_dataset("dxi", data=dxi)  # (grid_densities_calculated_on)
        f.create_dataset("targets", data=targets)  # (x_target locations)

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


class DensityMaker:
    def __init__(self, xc, yc, model, dxi):
        self.xc = xc
        self.yc = yc
        self.model = model
        self.dxi = dxi

    def generate(self, target_x):
        xt = torch.tensor([target_x]).reshape(1, 1, -1)
        xt_ag = nps.AggregateInput((xt, 0))
        # TODO: Find out how to do this while setting the seed for reproducibility
        # Could do this in a batched fashion, I think
        pred = self.model(self.xc, self.yc, xt_ag)
        ad0 = plot_densities(pred, self.dxi)
        return ad0


def clean_config(config: dict) -> dict:
    if "max_len" not in config:
        config["max_len"] = None
    if "num_samples" not in config:
        config["num_samples"] = None
    config["model_weights"] = Path(config["model_weights"])
    return config


def generate_samples(config: dict, out_samples: Path, overwrite=False):
    model = load_model(config["model_weights"], name=config["name"])

    contexts = to_contexts(config["contexts"])
    gen = construct_trajectory_gens(
        trajectory_length=config["trajectory"]["length"],
        x_range=(config["trajectory"]["low"], config["trajectory"]["high"]),
        x_context=contexts[0][0],
    )[config["trajectory"]["generator"]]
    # Move generator dictionary into SampleSet? That way we can reconstruct
    # May also want metadata about what model is made, so just write all config to
    # metadata for SampleSet?
    ss = SampleSet(out_samples, contexts, gen, overwrite=overwrite)
    ss.create_samples(model, config["n_mixtures"])
    return ss


def get_dxi_and_targets(config):
    dr = config["density"]["range"]
    tr = config["targets"]["range"]
    dxi = torch.arange(dr["start"], dr["end"], dr["step"])
    targets = torch.arange(tr["start"], tr["end"], tr["step"])
    return dxi, targets


def generate_densities(
    config: dict,
    in_samples: Path,
    out_marginal_densities: Path,
    overwrite=False,
    workers=None,
):
    model = load_model(config["model_weights"], config["name"])  # this has determined config, not ideal
    # TODO: make sure contexts are the same as in the samples?
    # Load directly from the ss instead?
    ss = read_hdf5(in_samples)  # overwrites existing
    dxi, targets = get_dxi_and_targets(config)

    if out_marginal_densities.exists() and not overwrite:
        LOG.warning(f"{out_marginal_densities} file already exists!")
    else:
        generate_marginal_densities(
            out_marginal_densities,
            ss,
            model,
            targets,
            dxi,
            config["max_len"],
            workers,
        )


def load_model(weights, name):
    print("Name:", name)
    if name in ["sawtooth", "synthetic-audio", "simple-mixture"]:
        model = get_model_gen()
    elif name in ["real-audio"]:
        model = get_model_real_audio()
    else:
        raise ValueError(f"Unknown model name: {name}")
    model.load_state_dict(torch.load(weights, map_location="cpu")["weights"])
    return model


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
    model = nps.construct_convgnp(
        points_per_unit=1,
        dim_x=1,
        dim_yc=(1,) * 1,
        dim_yt=1,
        likelihood="het",
        # conv_arch=args.arch,
        unet_channels=(128,) * 6,
        unet_strides=(1, 2, 2, 2, 2, 2),
        conv_channels=64,
        conv_layers=6,
        conv_receptive_field=50,
        margin=0.25, # 07_10
        # margin=1,  # 07_8
        encoder_scales=None,
        transform=None,
    )
    return model


def get_context():
    gen = nps.SawtoothGenerator(
        torch.float32,
        seed=2,
        batch_size=1,  # Only need one sample.
        # Use only two context points to introduce ambiguity.
        num_context=nps.UniformDiscrete(1, 1),
        # Be sure to use the same distribution of frequencies we used during training.
        dist_freq=nps.UniformContinuous(2, 4),
        num_target=nps.UniformDiscrete(50, 50),
    )
    batch = gen.generate_batch()  # Sample a batch of data.
    contexts = batch["contexts"]
    return contexts


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
    m2 = pred.mean.elements[0].detach().numpy().reshape(-1)
    v2 = pred.var.elements[0].detach().numpy().reshape(-1)

    densities = np.zeros(dxi.shape)
    # TODO: should be faster way to do this
    # parallelize here instead?

    for m, v in zip(m2, v2):
        densities += norm.pdf(dxi, loc=m, scale=np.sqrt(v))
    densities = densities / len(m2)
    return densities


def to_contexts(
    context_list: List[List[float]],
) -> List[Tuple[torch.Tensor, torch.Tensor]]:
    try:
        x, y = zip(*context_list)
        contexts = [
            (torch.Tensor(x).reshape(1, 1, -1), torch.Tensor(y).reshape(1, 1, -1))]
    # TODO: probably could make this cleaner instead of catching exception
    # Error is the context is empty
    except ValueError:
        contexts = [(torch.Tensor([]).reshape(1, 1, -1), torch.Tensor([]).reshape(1, 1, -1))]
    return contexts


def main(
    config_path: Path,
    out_samples: Path,
    out_densities: Path,
    overwrite: bool = False,
    workers=None,
):
    with open(config_path, "r") as f:
        config = clean_config(yaml.safe_load(f))
    LOG.info(f"Generating samples with config: {config}")
    generate_samples(config, out_samples, overwrite)
    LOG.info(f"Generating densities with config: {config}")
    generate_densities(config, out_samples, out_densities, overwrite, workers)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Say hello")
    parser.add_argument("c", help="Path to AR configuration", type=Path)
    parser.add_argument("s", help="Path to output samples hdf5", type=Path)
    parser.add_argument("d", help="Path to output marginal densities hdf5", type=Path)
    parser.add_argument("--overwrite", dest="overwrite", action="store_true")
    parser.add_argument("--no-overwrite", dest="overwrite", action="store_false")
    parser.set_defaults(overwrite=False)
    parser.add_argument(
        "--workers",
        help="number of workers to use. Defaults to None",
        type=str,
    )

    args = parser.parse_args()
    if args.workers == "half":
        workers = "half"
    elif args.workers is not None:
        workers = int(args.workers)
    else:
        workers = None

    main(args.c, args.s, args.d, args.overwrite, workers)
