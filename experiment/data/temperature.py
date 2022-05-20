import neuralprocesses.torch as nps
import torch

from .util import register_data

__all__ = []


def setup(args, config, *, num_tasks_train, num_tasks_cv, num_tasks_eval, device):
    config["dim_x"] = 2
    config["dim_y"] = 1

    if args.model == "convcnp-mlp":
        config["model"] = nps.construct_climate_convgnp_mlp(likelihood="het")
        context_sample = False
        target_elev = True
        target_square = 0
        do_plot = False
    elif args.model == "convgnp-mlp":
        config["model"] = nps.construct_climate_convgnp_mlp(likelihood="lowrank")
        context_sample = False
        target_elev = True
        target_square = 0
        do_plot = False
    elif args.model == "convcnp-multires":
        config["model"] = nps.construct_climate_convgnp_multires(likelihood="het")
        context_sample = True
        target_elev = False
        target_square = 3
        do_plot = True
    elif args.model == "convgnp-multires":
        config["model"] = nps.construct_climate_convgnp_multires(likelihood="lowrank")
        context_sample = True
        target_elev = False
        target_square = 3
        do_plot = True
    else:
        raise ValueError(f'Experiment does not yet support model "{args.model}".')

    # Other settings specific to the temperature experiment:
    if do_plot:
        config["plot"] = {2: {"range": ((6, 16), (47, 55))}}
    else:
        config["plot"] = {}

    gen_train = nps.TemperatureGenerator(
        torch.float32,
        seed=10,
        batch_size=args.batch_size,
        context_sample=context_sample,
        context_sample_factor=10,
        target_min=10,
        target_square=target_square,
        target_elev=target_elev,
        subset="train",
        device=device,
    )
    gen_cv = lambda: nps.TemperatureGenerator(
        torch.float32,
        seed=20,
        batch_size=args.batch_size,
        context_sample=context_sample,
        context_sample_factor=10,
        target_min=1,
        target_square=target_square,
        target_elev=target_elev,
        subset="cv",
        # Cycle over the data a few times to account for the random square sampling.
        passes=10,
        device=device,
    )
    gens_eval = lambda: [
        (
            "Downscaling",
            nps.TemperatureGenerator(
                torch.float32,
                seed=30,
                batch_size=args.batch_size,
                context_sample=False,
                target_min=1,
                target_square=2 if "eval-square" in args.experiment_setting else 0,
                target_elev=target_elev,
                subset="eval",
                device=device,
            ),
        ),
        (
            "Fusion",
            nps.TemperatureGenerator(
                torch.float32,
                seed=30,
                batch_size=args.batch_size,
                context_sample=True,
                context_sample_factor=10,
                target_min=1,
                target_square=2 if "eval-square" in args.experiment_setting else 0,
                target_elev=target_elev,
                subset="eval",
                device=device,
            ),
        ),
    ]
    return gen_train, gen_cv, gens_eval


register_data("temperature", setup)
