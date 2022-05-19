import torch

import neuralprocesses.torch as nps
from .util import register_data

__all__ = []


def setup(args, config, *, num_tasks_train, num_tasks_cv, num_tasks_eval, device):
    config["dim_x"] = 1
    config["dim_y"] = 2

    # Architecture choices specific for the predator-prey experiments:
    config["points_per_unit"] = 4
    config["margin"] = 1
    config["dws_receptive_field"] = 100
    config["transform"] = "softplus"

    # Other settings specific to the predator-prey experiments:
    config["plot"] = {1: {"range": (0, 100), "axvline": []}}

    gen_train = nps.PredPreyGenerator(
        torch.float32,
        seed=10,
        batch_size=args.batch_size,
        num_tasks=num_tasks_train,
        device=device,
    )
    gen_cv = lambda: nps.PredPreyGenerator(
        torch.float32,
        seed=20,
        batch_size=args.batch_size,
        num_tasks=num_tasks_cv,
        device=device,
    )
    gens_eval = lambda: (
        "Evaluation",
        nps.PredPreyGenerator(
            torch.float32,
            seed=30,
            batch_size=args.batch_size,
            num_tasks=num_tasks_eval,
            device=device,
        ),
    )
    return gen_train, gen_cv, gens_eval


register_data("predprey", setup)
