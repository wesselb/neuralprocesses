import argparse
import logging
import os
from pathlib import Path
from typing import List, Tuple

import experiment as exp
import torch
import yaml
from collections import namedtuple

import neuralprocesses.torch as nps
from neuralprocesses.model.sampler import SampleSet, read_hdf5, load_model, \
    generate_marginal_densities, get_dxi_and_targets, clean_config
from neuralprocesses.model.trajectory import construct_trajectory_gens

LOGLEVEL = os.environ.get("LOGLEVEL", "WARNING").upper()
logging.basicConfig(level=LOGLEVEL)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)


def get_batch(generator_kwargs):
    ARGS = namedtuple("args", generator_kwargs)
    args = ARGS(**generator_kwargs)
    config = {}

    gen_train, gen_cv, gens_eval = exp.data[args.data]["setup"](
        args,
        config,
        num_tasks_train=2**6,
        num_tasks_cv=2**6,
        num_tasks_eval=2**6,
        device="cpu",
    )
    gen = gens_eval
    batch = gen.generate_batch()
    return batch


def generate_samples(config: dict, out_samples: Path, overwrite=False):
    model = load_model(config["model_weights"], name=config["name"])
    batch = get_batch(config["generator_kwargs"])
    contexts = batch["contexts"]

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
