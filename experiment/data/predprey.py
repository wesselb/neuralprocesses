import torch

import neuralprocesses.torch as nps
from .util import register_data

__all__ = []


def setup(args, config, *, num_tasks_train, num_tasks_cv, num_tasks_eval, device):
    config["default"]["rate"] = 1e-4
    config["default"]["epochs"] = 200
    config["dim_x"] = 1
    config["dim_y"] = 2

    # Architecture choices specific for the predator-prey experiments:
    config["transform"] = "softplus"

    # Configure the convolutional models:
    config["points_per_unit"] = 4
    config["margin"] = 1
    config["conv_receptive_field"] = 100
    config["unet_strides"] = (1,) + (2,) * 6
    config["unet_channels"] = (64,) * 7

    # Other settings specific to the predator-prey experiments:
    config["plot"] = {1: {"range": (0, 100), "axvline": []}}

    gen_train = nps.PredPreyGenerator(
        torch.float32,
        seed=10,
        batch_size=args.batch_size,
        num_tasks=num_tasks_train,
        mode="random",
        device=device,
    )
    gen_cv = lambda: nps.PredPreyGenerator(
        torch.float32,
        seed=20,
        batch_size=args.batch_size,
        num_tasks=num_tasks_cv,
        mode="random",
        device=device,
    )

    def gens_eval():
        return [
            # For the real tasks, the batch size will be one. Keep the number of batches
            # the same.
            (
                "Interpolation (Simulated)",
                nps.PredPreyGenerator(
                    torch.float32,
                    seed=30,
                    batch_size=args.batch_size,
                    num_tasks=num_tasks_eval,
                    mode="interpolation",
                    device=device,
                ),
            ),
            (
                "Forecasting (Simulated)",
                nps.PredPreyGenerator(
                    torch.float32,
                    seed=30,
                    batch_size=args.batch_size,
                    num_tasks=num_tasks_eval,
                    mode="forecasting",
                    device=device,
                ),
            ),
            (
                "Reconstruction (Simulated)",
                nps.PredPreyGenerator(
                    torch.float32,
                    seed=30,
                    batch_size=args.batch_size,
                    num_tasks=num_tasks_eval,
                    mode="reconstruction",
                    device=device,
                ),
            ),
            # For the real tasks, the batch size will be one. Keep the number of batches
            # the same.
            (
                "Interpolation (Real)",
                nps.PredPreyRealGenerator(
                    torch.float32,
                    seed=30,
                    num_tasks=num_tasks_eval // args.batch_size,
                    mode="interpolation",
                    device=device,
                ),
            ),
            (
                "Forecasting (Real)",
                nps.PredPreyRealGenerator(
                    torch.float32,
                    seed=30,
                    num_tasks=num_tasks_eval // args.batch_size,
                    mode="forecasting",
                    device=device,
                ),
            ),
            (
                "Reconstruction (Real)",
                nps.PredPreyRealGenerator(
                    torch.float32,
                    seed=30,
                    num_tasks=num_tasks_eval // args.batch_size,
                    mode="reconstruction",
                    device=device,
                ),
            ),
        ]

    return gen_train, gen_cv, gens_eval


register_data("predprey", setup)
