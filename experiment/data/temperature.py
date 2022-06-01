from functools import partial

import torch

import neuralprocesses.torch as nps
from .util import register_data

__all__ = []


def setup(
    data_task,
    data_fold,
    args,
    config,
    *,
    num_tasks_train,
    num_tasks_cv,
    num_tasks_eval,
    device,
):
    config["dim_x"] = 2
    config["dim_y"] = 1

    if data_task == "germany":
        context_elev_hr = True
        data_task_train = "germany"
        data_task_cv = "germany"
        data_task_eval = "germany"
        lr_deg = 0.75
    elif data_task == "europe":
        context_elev_hr = False
        data_task_train = "europe"
        data_task_cv = "europe"
        data_task_eval = "value"
        lr_deg = 2
    elif data_task == "value":
        context_elev_hr = False
        data_task_train = "value"
        data_task_cv = "value"
        data_task_eval = "value"
        lr_deg = 2
    else:
        raise ValueError(f'Bad task "{data_task}".')

    if args.model in {"convcnp-mlp", "convgnp-mlp"}:
        if args.model == "convcnp-mlp":
            likelihood = "het"
        elif args.model == "convgnp-mlp":
            likelihood = "lowrank"
            # Help the mean a little in the beginning.
            config["fix_noise"] = True
            config["fix_noise_epochs"] = 10
        else:
            raise RuntimeError("Could not determine likelihood.")
        config["model"] = nps.construct_climate_convgnp_mlp(
            lr_deg=lr_deg,
            likelihood=likelihood,
        )
        context_sample = False
        target_elev = True
        target_square = 0
        do_plot = False

        # Set defaults.
        config["default"]["rate"] = 2.5e-5
        config["default"]["epochs"] = 500

    elif args.model in {"convcnp-multires", "convgnp-multires"}:
        if args.model == "convcnp-multires":
            likelihood = "het"
        elif args.model == "convgnp-multires":
            likelihood = "lowrank"
        else:
            raise RuntimeError("Could not determine likelihood.")
        config["model"] = nps.construct_climate_convgnp_multires(
            lr_deg=lr_deg,
            likelihood=likelihood,
        )
        context_sample = True
        target_elev = False
        target_square = 3
        do_plot = True

        # Set defaults.
        config["default"]["rate"] = 1e-5
        config["default"]["epochs"] = 500
        config["default"]["also_ar"] = True

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
        context_sample_factor=1000,
        context_sample_factor_at=600,
        context_elev_hr=context_elev_hr,
        target_min=10,
        target_square=target_square,
        target_elev=target_elev,
        subset="train",
        data_task=data_task_train,
        data_fold=data_fold,
        device=device,
    )
    gen_cv = lambda: nps.TemperatureGenerator(
        torch.float32,
        seed=20,
        batch_size=args.batch_size,
        context_sample=context_sample,
        context_sample_factor=1000,
        context_sample_factor_at=600,
        context_elev_hr=context_elev_hr,
        target_min=1,
        target_square=target_square,
        target_elev=target_elev,
        subset="cv",
        data_task=data_task_cv,
        data_fold=data_fold,
        # Cycle over the data a few times to account for the random square sampling.
        passes=10 if target_square > 0 else 1,
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
                context_elev_hr=context_elev_hr,
                target_min=1,
                target_square=3 if "eval-square" in args.experiment_setting else 0,
                target_elev=target_elev,
                subset="eval",
                data_task=data_task_eval,
                data_fold=data_fold,
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
                context_sample_factor=1000,
                context_sample_factor_at=600,
                context_elev_hr=context_elev_hr,
                target_min=1,
                target_square=3 if "eval-square" in args.experiment_setting else 0,
                target_elev=target_elev,
                subset="eval",
                data_task=data_task_eval,
                data_fold=data_fold,
                device=device,
            ),
        ),
    ]
    return gen_train, gen_cv, gens_eval


for i in [1, 2, 3, 4, 5]:
    register_data(f"temperature-germany-{i}", partial(setup, "germany", i))
    register_data(f"temperature-europe-{i}", partial(setup, "europe", i))
    register_data(f"temperature-value-{i}", partial(setup, "value", i))
