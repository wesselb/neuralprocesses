import argparse
import os
import shutil
import sys
import time
import warnings
from functools import partial

from tensorboardX import SummaryWriter

import experiment as exp
import lab as B
import neuralprocesses.torch as nps
import numpy as np
import torch
import wbml.out as out
from matrix.util import ToDenseWarning
from wbml.experiment import WorkingDirectory

__all__ = ["main"]

# Commands to run:
# python train.py --model dpconvcnp --data eq --min-scale 0.25 --max-scale 0.25 --encoder-scales 0.2 --batch-size 16 --dp-epsilon-min 3 --dp-epsilon-max 3 --epochs 1000 --prefix batch-size-16
# python train.py --model dpconvcnp --data eq --min-scale 0.25 --max-scale 0.25 --encoder-scales 0.2 --batch-size 8 --dp-epsilon-min 3 --dp-epsilon-max 3 --epochs 1000 --prefix batch-size-8

warnings.filterwarnings("ignore", category=ToDenseWarning)

def split_in_two(batch):

    batch_size = batch["contexts"][0][0].shape[0]

    batch_1 = {}
    batch_2 = {}

    batch_1["contexts"] = [
        (batch["contexts"][0][0][:batch_size//2], batch["contexts"][0][1][:batch_size//2]),
    ]
    batch_1["xt"] = batch["xt"][:batch_size//2]
    batch_1["yt"] = batch["yt"][:batch_size//2]
    batch_1["epsilon"] = batch["epsilon"][:batch_size//2]
    batch_1["delta"] = batch["delta"][:batch_size//2]
    

    batch_2["contexts"] = [
        (batch["contexts"][0][0][batch_size//2:], batch["contexts"][0][1][batch_size//2:]),
    ]
    batch_2["xt"] = batch["xt"][batch_size//2:]
    batch_2["yt"] = batch["yt"][batch_size//2:]
    batch_2["epsilon"] = batch["epsilon"][batch_size//2:]
    batch_2["delta"] = batch["delta"][batch_size//2:]

    assert batch_1["contexts"][0][0].shape == batch_2["contexts"][0][0].shape
    assert batch_1["contexts"][0][1].shape == batch_2["contexts"][0][1].shape
    assert batch_1["xt"].shape == batch_2["xt"].shape
    assert batch_1["yt"].shape == batch_2["yt"].shape
    
    return batch_1, batch_2



def train(state, model, opt, objective, gen, *, fix_noise, epoch, step, summary_writer, num_forward=1):
    """Train for an epoch."""

    NUM_STEPS_PER_GRAD_BIN = 1000
    scale_param_grads = []

    vals = []
    for _batch in gen.epoch():
        for batch in split_in_two(_batch):
            opt.zero_grad(set_to_none=True)
            losses_logging = []
            for _ in range(num_forward):
                state, forward_obj = objective(
                    state,
                    model,
                    batch["contexts"],
                    batch["xt"],
                    batch["yt"],
                    epsilon=batch["epsilon"],
                    delta=batch["delta"],
                    fix_noise=fix_noise,
                )
                forward_obj = - B.mean(forward_obj / num_forward)
                forward_obj.backward()
                vals.append(B.to_numpy(forward_obj)[None])
                losses_logging.append(B.to_numpy(forward_obj))
            #out.kv("Encoder grad scale       ", model.encoder.coder[1][0].log_scale.grad)
            opt.step()

            scale_param_grad = model.encoder.coder[2][0]._log_scale.grad
            scale_param_grads.append(scale_param_grad)

            summary_writer.add_scalar("train_step_loss", np.sum(losses_logging), step)
            summary_writer.add_scalar("train_step_scale", B.exp(model.encoder.coder[2][0].log_scale), step)
            summary_writer.add_scalar("train_step_scale_param_grad", scale_param_grad, step)

            if len(scale_param_grads) >= NUM_STEPS_PER_GRAD_BIN:
                summary_writer.add_scalar(
                    f"{NUM_STEPS_PER_GRAD_BIN}_step_scale_param_grad_var",
                    torch.var(torch.tensor(scale_param_grads)),
                    step,
                )
                summary_writer.add_scalar(
                    f"{NUM_STEPS_PER_GRAD_BIN}_step_scale_param_grad_stddev",
                    torch.var(torch.tensor(scale_param_grads))**0.5,
                    step,
                )
                summary_writer.add_scalar(
                    f"{NUM_STEPS_PER_GRAD_BIN}_step_scale_param_grad_mean",
                    torch.mean(torch.tensor(scale_param_grads)),
                    step,
                )
                scale_param_grads = []
            step = step + 1

    vals = B.concat(*vals)
    out.kv("Loglik (T)", exp.with_err(vals, and_lower=True))
    summary_writer.add_scalar("train_epoch_loglik", B.mean(vals), epoch)
    return state, B.mean(vals) - 1.96 * B.std(vals) / B.sqrt(len(vals)), step


def eval(state, model, objective, gen, *, epoch, summary_writer):
    """Perform evaluation."""
    with torch.no_grad():
        vals, kls, kls_diag = [], [], []
        for batch in gen.epoch():

            state, obj = objective(
                state,
                model,
                batch["contexts"],
                batch["xt"],
                batch["yt"],
                epsilon=batch["epsilon"],
                delta=batch["delta"],
            )

            # Save numbers.
            n = nps.num_data(batch["xt"], batch["yt"])
            vals.append(B.to_numpy(obj))
            if "pred_logpdf" in batch:
                kls.append(B.to_numpy(batch["pred_logpdf"] / n - obj))
            if "pred_logpdf_diag" in batch:
                kls_diag.append(B.to_numpy(batch["pred_logpdf_diag"] / n - obj))

        # Report numbers.
        vals = B.concat(*vals)
        metrics = {"loglik_val": exp.with_err(vals, and_lower=True)}
        out.kv("Loglik (V)", metrics["loglik_val"])
        if kls:
            metrics["kl_full"] = exp.with_err(B.concat(*kls), and_upper=True)
            out.kv("KL (full)", metrics["kl_full"])
        if kls_diag:
            metrics["kl_diag"] = exp.with_err(B.concat(*kls_diag), and_upper=True)
            out.kv("KL (diag)", metrics["kl_diag"])
            
        #out.kv("Encoder scale       ", torch.exp(model.encoder.coder[1][0].log_scale))
        summary_writer.add_scalar("val_epoch_loglik", np.mean(-vals), epoch)
        summary_writer.add_scalar("val_epoch_kl_full", np.mean(kls), epoch)
        summary_writer.add_scalar("val_epoch_kl_diag", np.mean(kls_diag), epoch)

        return state, B.mean(vals) - 1.96 * B.std(vals) / B.sqrt(len(vals)), metrics


def main(**kw_args):
    # Setup arguments.
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=str, nargs="*", default=["_experiments_debugging"])
    parser.add_argument("--subdir", type=str, nargs="*")
    parser.add_argument("--device", type=str)
    parser.add_argument("--gpu", type=int)
    parser.add_argument("--checkpoint-every", type=int, default=None)
    parser.add_argument("--dim-x", type=int, default=1)
    parser.add_argument("--dim-y", type=int, default=1)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--rate", type=float)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument(
        "--model",
        choices=[
            "cnp",
            "gnp",
            "np",
            "acnp",
            "agnp",
            "anp",
            "convcnp",
            "convgnp",
            "convnp",
            "fullconvgnp",
            # Experiment-specific architectures:
            "convcnp-mlp",
            "convgnp-mlp",
            "convcnp-multires",
            "convgnp-multires",
            "dpconvcnp",
        ],
        default="convcnp",
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
    parser.add_argument("--objective", choices=["loglik", "elbo"], default="loglik")
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
    parser.add_argument(
        "--eeg-mode",
        type=str,
        choices=["random", "interpolation", "forecasting", "reconstruction"],
    )
    parser.add_argument("--patch", type=str)
    parser.add_argument("--encoder-scales", type=float, default=None)

    parser.add_argument("--min-scale", type=float)
    parser.add_argument("--max-scale", type=float)

    parser.add_argument("--prefix", type=str, default=None)
    parser.add_argument("--num-forward", type=int, default=1)
    parser.add_argument("--dp-epsilon-min", type=float, default=1.)
    parser.add_argument("--dp-epsilon-max", type=float, default=9.)
    parser.add_argument("--dp-log10-delta-min", type=float, default=-3.)
    parser.add_argument("--dp-log10-delta-max", type=float, default=-3.)
    parser.add_argument("--dp-y-bound", type=float, default=2.)
    parser.add_argument("--dp-t", type=float, default=0.5)
    parser.add_argument("--dp-learn-params", default=False, action="store_true")
    parser.add_argument("--dp-amortise-params", default=False, action="store_true")
    parser.add_argument("--dp-use-noise-channels", default=False, action="store_true")

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

        dp_epsilon_range = (args.dp_epsilon_min, args.dp_epsilon_max)
        dp_log10_delta_range = (args.dp_log10_delta_min, args.dp_log10_delta_max)

        if args.dp_amortise_params:
            dp_param_prefix = "a_"

        elif args.dp_learn_params:
            dp_param_prefix = f"l-{args.dp_y_bound:.2f}-{args.dp_t}_"

        else:
            dp_param_prefix = f"f-{args.dp_y_bound:.2f}-{args.dp_t}_"

        model_name = args.prefix + "_dpconvcnp_"
        model_name = model_name + dp_param_prefix
        model_name = model_name + (f"nc_" if args.dp_use_noise_channels else "")
        model_name = model_name + f"s-{args.min_scale:.2f}-{args.max_scale:.2f}_"
        model_name = model_name + f"e-{dp_epsilon_range[0]:.0f}-{dp_epsilon_range[1]:.0f}_"
        model_name = model_name + f"d-{dp_log10_delta_range[0]:.0f}-{dp_log10_delta_range[1]:.0f}"

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

    # Create summary writer
    if os.path.exists(f"{wd.root}/metrics"):
        shutil.rmtree(f"{wd.root}/metrics")
    
    summary_writer = SummaryWriter(log_dir=f"{wd.root}/metrics")

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
        "encoder_scales": None or args.encoder_scales,
        "fullconvgnp_kernel_factor": 2,
        "mean_diff": args.mean_diff,
        # Performance of the ConvGNP is sensitive to this parameter. Moreover, it
        # doesn't make sense to set it to a value higher of the last hidden layer of
        # the CNN architecture. We therefore set it to 64.
        "num_basis_functions": 64,
        "eeg_mode": args.eeg_mode,
        "dp_epsilon_range": dp_epsilon_range,
        "dp_log10_delta_range": dp_log10_delta_range,
        "min_log10_scale": np.log10(args.min_scale),
        "max_log10_scale": np.log10(args.max_scale),
    }

    # Setup data generators for training and for evaluation.
    gen_train, gen_cv, gens_eval = exp.data[args.data]["setup"](
        args,
        config,
        num_tasks_train=2**6 if args.train_fast else 2**14,
        num_tasks_cv=2**6 if args.train_fast else 2**12,
        num_tasks_eval=2**6 if args.evaluate_fast else 2**14,
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
        elif args.model == "gnp":
            model = nps.construct_gnp(
                dim_x=config["dim_x"],
                dim_yc=(1,) * config["dim_y"],
                dim_yt=config["dim_y"],
                dim_embedding=config["dim_embedding"],
                enc_same=config["enc_same"],
                num_dec_layers=config["num_layers"],
                width=config["width"],
                likelihood="lowrank",
                num_basis_functions=config["num_basis_functions"],
                transform=config["transform"],
            )
        elif args.model == "np":
            model = nps.construct_gnp(
                dim_x=config["dim_x"],
                dim_yc=(1,) * config["dim_y"],
                dim_yt=config["dim_y"],
                dim_embedding=config["dim_embedding"],
                enc_same=config["enc_same"],
                num_dec_layers=config["num_layers"],
                width=config["width"],
                likelihood="het",
                dim_lv=config["dim_embedding"],
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
        elif args.model == "agnp":
            model = nps.construct_agnp(
                dim_x=config["dim_x"],
                dim_yc=(1,) * config["dim_y"],
                dim_yt=config["dim_y"],
                dim_embedding=config["dim_embedding"],
                enc_same=config["enc_same"],
                num_heads=config["num_heads"],
                num_dec_layers=config["num_layers"],
                width=config["width"],
                likelihood="lowrank",
                num_basis_functions=config["num_basis_functions"],
                transform=config["transform"],
            )
        elif args.model == "anp":
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
                dim_lv=config["dim_embedding"],
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
        elif args.model == "convgnp":
            model = nps.construct_convgnp(
                points_per_unit=config["points_per_unit"],
                dim_x=config["dim_x"],
                dim_yc=(1,) * config["dim_y"],
                dim_yt=config["dim_y"],
                likelihood="lowrank",
                conv_arch=args.arch,
                unet_channels=config["unet_channels"],
                unet_strides=config["unet_strides"],
                conv_channels=config["conv_channels"],
                conv_layers=config["num_layers"],
                conv_receptive_field=config["conv_receptive_field"],
                num_basis_functions=config["num_basis_functions"],
                margin=config["margin"],
                encoder_scales=config["encoder_scales"],
                transform=config["transform"],
            )
        elif args.model == "convnp":
            if config["dim_x"] == 2:
                # Reduce the number of channels in the conv. architectures by a factor
                # $\sqrt(2)$. This keeps the runtime in check and reduces the parameters
                # of the ConvNP to the number of parameters of the ConvCNP.
                config["unet_channels"] = tuple(
                    int(c / 2**0.5) for c in config["unet_channels"]
                )
                config["dws_channels"] = int(config["dws_channels"] / 2**0.5)
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
                dim_lv=16,
                margin=config["margin"],
                encoder_scales=config["encoder_scales"],
                transform=config["transform"],
            )
        elif args.model == "fullconvgnp":
            model = nps.construct_fullconvgnp(
                points_per_unit=config["points_per_unit"],
                dim_x=config["dim_x"],
                dim_yc=(1,) * config["dim_y"],
                dim_yt=config["dim_y"],
                conv_arch=args.arch,
                unet_channels=config["unet_channels"],
                unet_strides=config["unet_strides"],
                conv_channels=config["conv_channels"],
                conv_layers=config["num_layers"],
                conv_receptive_field=config["conv_receptive_field"],
                kernel_factor=config["fullconvgnp_kernel_factor"],
                margin=config["margin"],
                encoder_scales=config["encoder_scales"],
                transform=config["transform"],
            )
        elif args.model == "dpconvcnp":

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
                divide_by_density=False,
                use_dp=True,
                dp_learn_params=args.dp_learn_params,
                dp_amortise_params=args.dp_amortise_params,
                dp_use_noise_channels=args.dp_use_noise_channels,
                dp_y_bound=args.dp_y_bound,
                dp_t=args.dp_t,
            )
        else:
            raise ValueError(f'Invalid model "{args.model}".')

    # Settings specific for the model:
    if config["fix_noise"] is None:
        if args.model in {"np", "anp", "convnp"}:
            config["fix_noise"] = True
        else:
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

    # Setup training objective.
    if args.objective == "loglik":
        objective = partial(
            nps.loglik,
            num_samples=args.num_samples,
            normalise=not args.unnormalised,
        )
        objective_cv = partial(
            nps.loglik,
            num_samples=args.num_samples,
            normalise=not args.unnormalised,
        )
        objectives_eval = [
            (
                "Loglik",
                partial(
                    nps.loglik,
                    num_samples=args.evaluate_num_samples,
                    batch_size=args.evaluate_batch_size,
                    normalise=not args.unnormalised,
                ),
            )
        ]
    elif args.objective == "elbo":
        objective = partial(
            nps.elbo,
            num_samples=args.num_samples,
            subsume_context=True,
            normalise=not args.unnormalised,
        )
        objective_cv = partial(
            nps.elbo,
            num_samples=args.num_samples,
            subsume_context=False,  # Lower bound the right quantity.
            normalise=not args.unnormalised,
        )
        objectives_eval = [
            (
                "ELBO",
                partial(
                    nps.elbo,
                    # Don't need a high number of samples, because it is unbiased.
                    num_samples=5,
                    subsume_context=False,  # Lower bound the right quantity.
                    normalise=not args.unnormalised,
                ),
            ),
            (
                "Loglik",
                partial(
                    nps.loglik,
                    num_samples=args.evaluate_num_samples,
                    batch_size=args.evaluate_batch_size,
                    normalise=not args.unnormalised,
                ),
            ),
        ]
    else:
        raise RuntimeError(f'Invalid objective "{args.objective}".')

    # See if the point was to just load everything.
    if args.load:
        return {
            "wd": wd,
            "gen_train": gen_train,
            "gen_cv": gen_cv,
            "gens_eval": gens_eval,
            "model": model,
        }

    # The user can just want to see some statistics about the model.
    if args.no_action:
        exit()

    if args.evaluate:
        # Perform evaluation.
        if args.evaluate_last:
            name = "model-last.torch"
        else:
            name = "model-best.torch"
        model.load_state_dict(
            patch_model(torch.load(wd.file(name), map_location=device))["weights"]
        )

        if not args.ar or args.also_ar:
            # Make some plots.
            gen = gen_cv()
            for i in range(args.evaluate_num_plots):
                exp.visualise(
                    model,
                    gen,
                    path=wd.file(f"evaluate-{i + 1:03d}.pdf"),
                    config=config,
                )

            # For every objective and evaluation generator, do the evaluation.
            for objecive_name, objective_eval in objectives_eval:
                with out.Section(objecive_name):
                    for gen_name, gen in gens_eval():
                        with out.Section(gen_name.capitalize()):
                            state, _, metrics = eval(state, model, objective_eval, gen)

        # Always run AR evaluation for the conditional models.
        if not args.no_ar and (
            args.model in {"cnp", "acnp", "convcnp"} or args.ar or args.also_ar
        ):
            # Make some plots.
            gen = gen_cv()
            for i in range(args.evaluate_num_plots):
                exp.visualise(
                    model,
                    gen,
                    path=wd.file(f"evaluate-ar-{i + 1:03d}.pdf"),
                    config=config,
                    predict=nps.ar_predict,
                )

            with out.Section("AR"):
                for name, gen in gens_eval():
                    with out.Section(name.capitalize()):
                        state, _, metrics = eval(
                            state,
                            model,
                            partial(
                                nps.ar_loglik,
                                order="random",
                                normalise=not args.unnormalised,
                            ),
                            gen,
                        )

        # Sleep for sixty seconds before exiting.
        out.out("Finished evaluation. Sleeping for a minute before exiting.")
        time.sleep(60)
    else:
        # Perform training. First, check if we want to resume training.
        start = 0
        if args.resume_at_epoch:
            start = args.resume_at_epoch - 1
            d_last = patch_model(
                torch.load(wd.file("model-last.torch"), map_location=device)
            )
            d_best = patch_model(
                torch.load(wd.file("model-best.torch"), map_location=device)
            )
            model.load_state_dict(d_last["weights"])
            best_eval_lik = d_best["objective"]
        else:
            best_eval_lik = -np.inf

        # Setup training loop.
        opt = torch.optim.Adam(model.parameters(), lr=args.rate)

        # Set regularisation high for the first epochs.
        original_epsilon = B.epsilon
        B.epsilon = config["epsilon_start"]

        step = 0

        for i in range(start, args.epochs):
            with out.Section(f"Epoch {i + 1}"):
                # Set regularisation to normal after the first epoch.
                if i > 0:
                    B.epsilon = original_epsilon

                # Checkpoint at regular intervals if specified
                if args.checkpoint_every is not None and i % args.checkpoint_every == 0:
                    out.out("Checkpointing...")
                    torch.save(
                        {
                            "weights": model.state_dict(),
                            "epoch": i + 1,
                        },
                        wd.file(f"model-epoch-{i+1}.torch"),
                    )

                # Perform an epoch.
                if config["fix_noise"] and i < config["fix_noise_epochs"]:
                    fix_noise = 1e-4
                else:
                    fix_noise = None
                state, _, step = train(
                    state,
                    model,
                    opt,
                    objective,
                    gen_train,
                    fix_noise=fix_noise,
                    epoch=i,
                    step=step,
                    summary_writer=summary_writer,
                    num_forward=args.num_forward,
                )

                # The epoch is done. Now evaluate.
                state, val, metrics = eval(state, model, objective_cv, gen_cv(), epoch=i, summary_writer=summary_writer)

                # Save current model.
                torch.save(
                    {
                        "weights": model.state_dict(),
                        "objective": val,
                        "epoch": i + 1,
                    },
                    wd.file(f"model-last.torch"),
                )

                # Check if the model is the new best. If so, save it.
                if val > best_eval_lik:
                    out.out("New best model!")
                    best_eval_lik = val
                    torch.save(
                        {
                            "weights": model.state_dict(),
                            "objective": val,
                            "epoch": i + 1,
                        },
                        wd.file(f"model-best.torch"),
                    )
                    
                    for metric_name, metric_value in metrics.items():
                        file = open(f"{wd.file(metric_name)}.txt", "w")
                        file.write(metric_value)
                        file.close()

                # Visualise a few predictions by the model.
                gen = gen_cv()
                for j in range(5):
                    exp.visualise(
                        model,
                        gen,
                        path=wd.file(f"train-epoch-{i + 1:03d}-{j + 1}.pdf"),
                        config=config,
                    )


if __name__ == "__main__":
    main()
