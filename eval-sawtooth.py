import argparse
import os
import sys
import time
import warnings
from functools import partial

import experiment as exp
import lab as B
import neuralprocesses.torch as nps
import numpy as np
import torch
import wbml.out as out
from matrix.util import ToDenseWarning
from wbml.experiment import WorkingDirectory

__all__ = ["main"]

warnings.filterwarnings("ignore", category=ToDenseWarning)


def eval(model, objective, gen, num_seeds):
    """Perform evaluation."""
    objectives = []
    with torch.no_grad():
        for i, batch in enumerate(gen.epoch()):
            objectives.append([])
            for seed in range(num_seeds):

                ar_state = B.create_random_state(torch.float32, seed=seed)

                _, obj = objective(
                    ar_state,
                    model,
                    batch["contexts"],
                    batch["xt"],
                    batch["yt"],
                )
                
                objectives[-1].append(obj)

            objectives[-1] = torch.stack(objectives[-1], axis=0)

    objectives = torch.stack(objectives, axis=0).detach().cpu().numpy()

    return objectives


def main(**kw_args):

    # Setup arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, nargs="*", default=["_experiments"])
    parser.add_argument("--subdir", type=str, nargs="*")
    parser.add_argument("--device", type=str)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--dim-x", type=int, default=1)
    parser.add_argument("--dim-y", type=int, default=1)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--rate", type=float)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument(
        "--model",
        choices=[
            "cnp",
            "acnp",
            "convcnp",
        ],
    )
    parser.add_argument(
        "--arch",
        choices=[
            "unet",
            "unet-sep",
            "unet-res",
            "unet-res-sep",
            "conv",
            "conv-sep",
            "conv-res",
            "conv-res-sep",
        ],
        default="unet",
    )
    parser.add_argument(
        "--data",
        choices=exp.data,
        default="eq",
    )
    parser.add_argument("--mean-diff", type=float, default=None)
    parser.add_argument("--objective", choices=["loglik"], default="loglik")
    parser.add_argument("--num-samples", type=int, default=20)
    parser.add_argument("--resume-at-epoch", type=int)
    parser.add_argument("--train-fast", action="store_true")
    parser.add_argument("--check-completed", action="store_true")
    parser.add_argument("--unnormalised", action="store_true")
    parser.add_argument("--evaluate", action="store_true")
    parser.add_argument("--evaluate-last", action="store_true")
    parser.add_argument("--evaluate-fast", action="store_true")
    parser.add_argument("--evaluate-num-plots", type=int, default=5)
    parser.add_argument(
        "--evaluate-objective",
        choices=["loglik", "elbo"],
        default="loglik",
    )
    parser.add_argument("--evaluate-num-samples", type=int, default=512)
    parser.add_argument("--evaluate-batch-size", type=int, default=8)
    parser.add_argument("--no-action", action="store_true")
    parser.add_argument("--load", action="store_true")
    parser.add_argument("--ar", action="store_true")
    parser.add_argument("--also-ar", action="store_true")
    parser.add_argument("--no-ar", action="store_true")
    parser.add_argument("--experiment-setting", type=str, nargs="*")
    parser.add_argument("--num-ar-seeds", type=int)
    parser.add_argument("--num-target", type=int)
    parser.add_argument(
        "--eeg-mode",
        type=str,
        choices=["random", "interpolation", "forecasting", "reconstruction"],
    )
    parser.add_argument("--patch", type=str)

    if kw_args:
        # Load the arguments from the keyword arguments passed to the function.
        # Carefully convert these to command line arguments.
        args = parser.parse_args(
            sum(
                [
                    ["--" + k.replace("_", "-")] + ([str(v)] if v is not True else [])
                    for k, v in kw_args.items()
                ],
                [],
            )
        )
    else:
        args = parser.parse_args()

    def patch_model(d):
        """Patch a loaded model.

        Args:
            d (dict): Output of :func:`torch.load`.

        Returns:
            dict: `d`, but patched.
        """
        if args.patch:
            with out.Section("Patching loaded model"):
                # Loop over patches.
                for patch in args.patch.strip().split(";"):
                    base_from, base_to = patch.split(":")

                    # Try to apply the patch.
                    applied_patch = False
                    for k in list(d["weights"].keys()):
                        if k.startswith(base_from):
                            applied_patch = True
                            tail = k[len(base_from) :]
                            d["weights"][base_to + tail] = d["weights"][k]
                            del d["weights"][k]

                    # Report whether the patch was applied.
                    if applied_patch:
                        out.out(f'Applied patch "{patch}".')
                    else:
                        out.out(f'Did not apply patch "{patch}".')
        return d

    # Remove the architecture argument if a model doesn't use it.
    if args.model not in {
        "convcnp",
        "convgnp",
        "convnp",
        "fullconvgnp",
        "dpconvcnp",
    }:
        del args.arch

    # Remove the dimensionality specification if the experiment doesn't need it.
    if not exp.data[args.data]["requires_dim_x"]:
        del args.dim_x
    if not exp.data[args.data]["requires_dim_y"]:
        del args.dim_y

    # Ensure that `args.experiment_setting` is always a list.
    if not args.experiment_setting:
        args.experiment_setting = []

    # Determine settings for the setup of the script.
    suffix = ""
    observe = False
    if args.check_completed or args.no_action or args.load:
        observe = True
    elif args.evaluate:
        suffix = "_evaluate"
        if args.ar:
            suffix += "_ar"
    else:
        # The default is training.
        suffix = "_train"

    data_dir = args.data if args.mean_diff is None else f"{args.data}-{args.mean_diff}"
    data_dir = data_dir if args.eeg_mode is None else f"{args.data}-{args.eeg_mode}"

    if args.model == "dpconvcnp":
        model_name = f"dpconvcnp_" + \
                     f"{args.dp_epsilon:.4f}_" + \
                     f"{args.dp_delta:.4f}_" + \
                     f"{args.dp_y_bound:.4f}_" + \
                     f"{args.dp_kernel_bound:.4f}"
                        
    else:
        model_name = args.model
    
    # Setup script.
    if not observe:
        out.report_time = True
    wd = WorkingDirectory(
        *args.root,
        *(args.subdir or ()),
        data_dir,
        *((f"x{args.dim_x}_y{args.dim_y}",) if hasattr(args, "dim_x") else ()),
        model_name,
        *((args.arch,) if hasattr(args, "arch") else ()),
        args.objective,
        log=f"log{suffix}.txt",
        diff=f"diff{suffix}.txt",
        observe=observe,
    )

    # Determine which device to use. Try to use a GPU if one is available.
    if args.device:
        device = args.device
    elif args.gpu is not None:
        device = f"cuda:{args.gpu}"
    elif torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    B.set_global_device(device)
    # Maintain an explicit random state through the execution.
    state = B.create_random_state(torch.float32, seed=0)

    # General config.
    config = {
        "default": {
            "epochs": None,
            "rate": None,
            "also_ar": False,
        },
        "epsilon": 1e-8,
        "epsilon_start": 1e-2,
        "cholesky_retry_factor": 1e6,
        "fix_noise": None,
        "fix_noise_epochs": 3,
        "width": 256,
        "dim_embedding": 256,
        "enc_same": False,
        "num_heads": 8,
        "num_layers": 6,
        "unet_channels": (64,) * 6,
        "unet_strides": (1,) + (2,) * 5,
        "conv_channels": 64,
        "encoder_scales": None,
        "fullconvgnp_kernel_factor": 2,
        "mean_diff": args.mean_diff,
        # Performance of the ConvGNP is sensitive to this parameter. Moreover, it
        # doesn't make sense to set it to a value higher of the last hidden layer of
        # the CNN architecture. We therefore set it to 64.
        "num_basis_functions": 64,
        "eeg_mode": args.eeg_mode,
    }

    # Setup data generators for training and for evaluation.
    gen_train, gen_cv, gens_eval = exp.data[args.data]["setup"](
        args,
        config,
        num_tasks_train=2**6 if args.train_fast else 2**14,
        num_tasks_cv=2**6 if args.train_fast else 2**12,
        num_tasks_eval=2**6 if args.evaluate_fast else 2**10, # 2**12,
        num_target=args.num_target,
        device=device,
    )

    # Apply defaults for the number of epochs and the learning rate. The experiment
    # is allowed to adjust these.
    args.epochs = args.epochs or config["default"]["epochs"] or 100
    args.rate = args.rate or config["default"]["rate"] or 3e-4
    args.also_ar = args.also_ar or config["default"]["also_ar"]

    # Check if a run has completed.
    if args.check_completed:
        if os.path.exists(wd.file("model-last.torch")):
            d = patch_model(torch.load(wd.file("model-last.torch"), map_location="cpu"))
            if d["epoch"] >= args.epochs - 1:
                out.out("Completed!")
                sys.exit(0)
        out.out("Not completed.")
        sys.exit(1)

    # Set the regularisation based on the experiment settings.
    B.epsilon = config["epsilon"]
    B.cholesky_retry_factor = config["cholesky_retry_factor"]

    if "model" in config:
        # See if the experiment constructed the particular flavour of the model already.
        model = config["model"]
    else:
        # Construct the model.
        if args.model == "cnp":
            model = nps.construct_gnp(
                dim_x=config["dim_x"],
                dim_yc=(1,) * config["dim_y"],
                dim_yt=config["dim_y"],
                dim_embedding=config["dim_embedding"],
                enc_same=config["enc_same"],
                num_dec_layers=config["num_layers"],
                width=config["width"],
                likelihood="het",
                transform=config["transform"],
            )
        elif args.model == "acnp":
            model = nps.construct_agnp(
                dim_x=config["dim_x"],
                dim_yc=(1,) * config["dim_y"],
                dim_yt=config["dim_y"],
                dim_embedding=config["dim_embedding"],
                enc_same=config["enc_same"],
                num_heads=config["num_heads"],
                num_dec_layers=config["num_layers"],
                width=config["width"],
                likelihood="het",
                transform=config["transform"],
            )
        elif args.model == "convcnp":
            model = nps.construct_convgnp(
                points_per_unit=config["points_per_unit"],
                dim_x=config["dim_x"],
                dim_yc=(1,) * config["dim_y"],
                dim_yt=config["dim_y"],
                likelihood="het",
                conv_arch=args.arch,
                unet_channels=config["unet_channels"],
                unet_strides=config["unet_strides"],
                conv_channels=config["conv_channels"],
                conv_layers=config["num_layers"],
                conv_receptive_field=config["conv_receptive_field"],
                margin=config["margin"],
                encoder_scales=config["encoder_scales"],
                transform=config["transform"],
            )
        else:
            raise ValueError(f'Invalid model "{args.model}".')

    config["fix_noise"] = False

    # Ensure that the model is on the GPU and print the setup.
    model = model.to(device)
    if not args.load:
        out.kv(
            "Arguments",
            {
                attr: getattr(args, attr)
                for attr in args.__dir__()
                if not attr.startswith("_")
            },
        )
        out.kv(
            "Config", {k: "<custom>" if k == "model" else v for k, v in config.items()}
        )
        out.kv("Number of parameters", nps.num_params(model))

    name = "model-last.torch"
    model.load_state_dict(
        patch_model(torch.load(wd.file(name), map_location=device))["weights"]
    )

    with out.Section("AR"):
        for short_name, name, gen in gens_eval():
            with out.Section(name.capitalize()):
                objectives = eval(
                    model,
                    partial(
                        nps.ar_loglik,
                        order="random",
                        normalise=not args.unnormalised,
                    ),
                    gen,
                    args.num_ar_seeds,
                )

                #with wd.file(f"{short_name}.txt") as f:
                #    f.write(f"{mean} +/- {stddev}")
                np.save(
                    wd.file(f"{short_name}-{args.num_target}.npy"),
                    objectives,
                )

    # Sleep for sixty seconds before exiting.
    out.out("Finished evaluation. Sleeping for a minute before exiting.")
    time.sleep(5)

if __name__ == "__main__":
    main()
