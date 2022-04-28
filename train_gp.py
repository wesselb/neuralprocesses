import argparse
import os
import sys
from functools import partial

import lab as B
import matplotlib.pyplot as plt
import numpy as np
import stheno
import torch
import wbml.out as out
from wbml.experiment import WorkingDirectory
from wbml.plot import tweak

import neuralprocesses.torch as nps


def train(state, model, objective, gen, *, epoch):
    """Train for an epoch."""
    vals = []
    for batch in gen.epoch():
        state, obj = objective(
            state,
            model,
            batch["xc"],
            batch["yc"],
            batch["xt"],
            batch["yt"],
            epoch=epoch,
        )
        vals.append(B.to_numpy(obj))
        # Be sure to negate the output of `objective`.
        val = -B.mean(obj)
        opt.zero_grad(set_to_none=True)
        val.backward()
        opt.step()
    out.kv("Loglik (T)", with_err(B.concat(*vals)))
    return state


def eval(state, model, objective, gen):
    """Perform evaluation."""
    with torch.no_grad():
        vals, kls, kls_diag = [], [], []
        for batch in gen.epoch():
            state, obj = objective(
                state,
                model,
                batch["xc"],
                batch["yc"],
                batch["xt"],
                batch["yt"],
            )

            # Save numbers.
            n = B.shape(batch["xt"], -1)
            vals.append(B.to_numpy(obj))
            if "pred_logpdf" in batch:
                kls.append(B.to_numpy(batch["pred_logpdf"] / n - obj))
            if "pred_logpdf_diag" in batch:
                kls_diag.append(B.to_numpy(batch["pred_logpdf_diag"] / n - obj))

        # Report numbers.
        out.kv("Loglik (V)", with_err(B.concat(*vals)))
        if kls:
            out.kv("KL (full)", with_err(B.concat(*kls)))
        if kls_diag:
            out.kv("KL (diag)", with_err(B.concat(*kls_diag)))

        return state, B.mean(B.concat(*vals))


def with_err(vals):
    """Print the mean value of a list of values with error."""
    vals = B.to_numpy(vals)
    mean = B.mean(vals)
    err = 1.96 * B.std(vals) / B.sqrt(B.length(vals))
    return f"{mean:8.4f} +- {err:8.4f}"


def visualise(model, gen, *, name, epoch, config, predict=nps.predict):
    """Plot the prediction for the first element of a batch."""
    if args.dim_x == 1:
        visualise_1d(
            model,
            gen,
            name=name,
            epoch=epoch,
            config=config[1],
            predict=predict,
        )
    elif args.dim_x == 2:
        visualise_2d(
            model,
            gen,
            name=name,
            epoch=epoch,
            config=config[2],
            predict=predict,
        )
    else:
        pass  # Not implemented. Just do nothing.


def visualise_1d(model, gen, *, name, epoch, config, predict):
    batch = gen.generate_batch()

    # Define points to predict at.
    with B.on_device(batch["xt"]):
        x = B.linspace(B.dtype(batch["xt"]), *config["range"], 500)

    # Predict with model and produce five noiseless samples.
    with torch.no_grad():
        mean, var, samples = predict(
            model,
            batch["xc"][:1, ...],
            batch["yc"][:1, ...],
            x[None, None, :],
        )

    plt.figure(figsize=(8, 6 * args.dim_y))

    for i in range(args.dim_y):
        plt.subplot(args.dim_y, 1, 1 + i)

        # Plot context and target.
        plt.scatter(
            batch["xc"][0, 0],
            batch["yc"][0, i],
            label="Context",
            style="train",
            s=20,
        )
        plt.scatter(
            batch["xt"][0, 0],
            batch["yt"][0, i],
            label="Target",
            style="test",
            s=20,
        )

        # Plot prediction.
        err = 1.96 * B.sqrt(var)
        plt.plot(
            x,
            mean[0, i],
            label="Prediction",
            style="pred",
        )
        plt.fill_between(
            x,
            mean[0, i] - err[0, i],
            mean[0, i] + err[0, i],
            style="pred",
        )
        plt.plot(
            x,
            samples[:, 0, i].T,
            style="pred",
            ls="-",
            lw=0.5,
        )

        # Plot prediction by ground truth.
        if hasattr(gen, "kernel") and args.dim_y == 1:
            f = stheno.GP(gen.kernel)
            # Make sure that everything is of `float64`s and on the GPU.
            noise = B.to_active_device(B.cast(torch.float64, gen.noise))
            xc = B.cast(torch.float64, batch["xc"][0, 0])
            yc = B.cast(torch.float64, batch["yc"][0, 0])
            x = B.cast(torch.float64, x)
            # Compute posterior GP.
            f_post = f | (f(xc, noise), yc)
            mean, lower, upper = f_post(x).marginal_credible_bounds()
            plt.plot(x, mean, label="Truth", style="pred2")
            plt.plot(x, lower, style="pred2")
            plt.plot(x, upper, style="pred2")

        for x_axvline in config["axvline"]:
            plt.axvline(x_axvline, c="k", ls="--", lw=0.5)
        plt.xlim(B.min(x), B.max(x))
        tweak()

    plt.savefig(wd.file(f"{name}-{epoch:03d}.pdf"))
    plt.close()


def visualise_2d(model, gen, *, name, epoch, config, predict):
    batch = gen.generate_batch()

    # Define points to predict at.
    with B.on_device(batch["xt"]):
        x = B.linspace(B.dtype(batch["xt"]), *config["range"], 100)[None, None, :]

    # Predict with model and produce five noiseless samples.
    with torch.no_grad():
        try:
            mean, _, samples = predict(
                model,
                batch["xc"][:1, ...],
                batch["yc"][:1, ...],
                (x, x),
                num_samples=2,
            )
        except:
            # The model probably doesn't suppose the tuple shorthand for grids. Do it
            # in a different way.
            with B.on_device(x):
                x0 = x[..., :, None]
                x1 = x[..., None, :]
                # Perform broadcasting.
                x0 = x0 * B.ones(x1)
                x1 = x1 * B.ones(x0)
                # Reshape into lists.
                x0 = B.reshape(x0, *B.shape(x0)[:-2], -1)
                x1 = B.reshape(x1, *B.shape(x1)[:-2], -1)
                # Run model on whole list.
                mean, _, samples = nps.predict(
                    model,
                    batch["xc"][:1, ...],
                    batch["yc"][:1, ...],
                    B.concat(x0, x1, axis=-2),
                    num_samples=2,
                )
                # Reshape the results back to images.
                mean = B.reshape(mean, *B.shape(mean)[:-1], 100, 100)
                samples = B.reshape(samples, *B.shape(samples)[:-1], 100, 100)

    vmin = max(B.max(mean), B.max(samples))
    vmax = min(B.min(mean), B.min(samples))

    def plot_imshow(image, i, label):
        plt.imshow(
            image.T,
            cmap="viridis",
            vmin=vmax,
            vmax=vmin,
            origin="lower",
            extent=[-2, 2, -2, 2],
            label=label,
        )
        plt.scatter(
            batch["xc"][0, 0],
            batch["xc"][0, 1],
            c=batch["yc"][0, i],
            cmap="viridis",
            vmin=vmax,
            vmax=vmin,
            edgecolor="white",
            linewidth=0.5,
            s=80,
            label="Context",
        )
        plt.scatter(
            batch["xt"][0, 0],
            batch["xt"][0, 1],
            c=batch["yt"][0, i],
            cmap="viridis",
            vmin=vmax,
            vmax=vmin,
            edgecolor="black",
            linewidth=0.5,
            s=80,
            marker="d",
            label="Target",
        )
        # Remove ticks, because those are noisy.
        plt.gca().set_xticks([])
        plt.gca().set_yticks([])

    plt.figure(figsize=(12, 4 * args.dim_y))

    for i in range(args.dim_y):
        # Plot mean.
        plt.subplot(args.dim_y, 3, 1 + i * 3)
        if i == 0:
            plt.title("Mean")
        plot_imshow(mean[0, i], i, label="Mean")
        tweak(grid=False)

        # Plot first sample.
        plt.subplot(args.dim_y, 3, 2 + i * 3)
        if i == 0:
            plt.title("Sample")
        plot_imshow(samples[0, 0, i], i, label="Sample 1")
        tweak(grid=False)

        # Plot second sample.
        plt.subplot(args.dim_y, 3, 3 + i * 3)
        if i == 0:
            plt.title("Sample")
        plot_imshow(samples[1, 0, i], i, label="Sample 2")
        tweak(grid=False)

    plt.savefig(wd.file(f"{name}-{epoch:03d}.pdf"))
    plt.close()


# Setup arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--root", type=str, nargs="*", default=["_experiments"])
parser.add_argument("--subdir", type=str, nargs="*")
parser.add_argument("--dim-x", type=int, default=1)
parser.add_argument("--dim-y", type=int, default=1)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--rate", type=float, default=3e-4)
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
    ],
    default="convcnp",
)
parser.add_argument("--arch", choices=["unet", "dws"], default="unet")
parser.add_argument(
    "--data",
    choices=[
        "eq",
        "matern",
        "weakly-periodic",
        "sawtooth",
        "mixture",
        "predprey",
    ],
    default="eq",
)
parser.add_argument("--objective", choices=["loglik", "elbo"], default="loglik")
parser.add_argument("--num-samples", type=int, default=20)
parser.add_argument("--resume-at-epoch", type=int)
parser.add_argument("--check-completed", action="store_true")
parser.add_argument("--evaluate", action="store_true")
parser.add_argument("--evaluate-last", action="store_true")
parser.add_argument("--evaluate-fast", action="store_true")
parser.add_argument("--evaluate-plot-num-samples", type=int, default=10)
parser.add_argument(
    "--evaluate-objective",
    choices=["loglik", "elbo"],
    default="loglik",
)
parser.add_argument("--evaluate-num-samples", type=int, default=4096)
parser.add_argument("--evaluate-batch-size", type=int, default=16)
parser.add_argument("--no-action", action="store_true")
parser.add_argument("--ar", action="store_true")
args = parser.parse_args()

# Remove the architecture argument if a model doesn't use it.
models_which_use_arch = {
    "convcnp",
    "convgnp",
    "convnp",
    "fullconvgnp",
}
if args.model not in models_which_use_arch:
    args.arch = None

# Determine the mode of the script.
if args.check_completed or args.no_action:
    # Don't add any mode suffix.
    mode = ""
elif args.evaluate:
    mode = "_evaluate"
    if args.ar:
        mode += "_ar"
else:
    # The default is training.
    mode = "_train"

# Setup script.
out.report_time = True
B.epsilon = 1e-8
wd = WorkingDirectory(
    *args.root,
    *(args.subdir or ()),
    args.data,
    f"x{args.dim_x}_y{args.dim_y}",
    args.model,
    *((args.arch,) if args.arch else ()),
    args.objective,
    log=f"log{mode}.txt",
    diff=f"diff{mode}.txt",
)

# Check if a run has completed.
if args.check_completed:
    # Simply check if the final plot exists.
    # TODO: Do this in a better way.
    if os.path.exists(wd.file(f"train-epoch-{args.epochs:03d}.pdf")):
        out.out("Completed!")
        sys.exit(0)
    else:
        out.out("Not completed.")
        sys.exit(1)

# Use a GPU if one is available.
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
B.set_global_device(device)
# Maintain an explicit random state through the execution.
state = B.create_random_state(torch.float32, seed=0)

# General architecture choices:
width = 256
dim_embedding = 256
num_heads = 8
num_layers = 6
unet_channels = (64,) * num_layers
dws_channels = 64
num_basis_functions = 512

# Setup data generators for training and for evaluation.
if args.data == "predprey":
    gen_train = nps.PredPreyGenerator(
        torch.float32,
        seed=10,
        batch_size=args.batch_size,
        num_tasks=2**14,
        x_ranges=((0, 100),) * args.dim_x,
        dim_y=args.dim_y,
    )
    gen_cv = nps.PredPreyGenerator(
        torch.float32,
        seed=20,
        batch_size=args.batch_size,
        num_tasks=2**12,
        x_ranges=((0, 100),) * args.dim_x,
        dim_y=args.dim_y,
    )
    gens_eval = lambda: (
        "Evaluation",
        nps.PredPreyGenerator(
            torch.float32,
            seed=30,
            batch_size=args.batch_size,
            num_tasks=2**6 if args.evaluate_fast else 2**14,
            x_ranges=((0, 100),) * args.dim_x,
            dim_y=args.dim_y,
        ),
    )

    # Architecture choices specific for the predator-prey experiments:
    points_per_unit = 4
    margin = 1
    dws_receptive_field = 100

    # Other settings specific to the predator-prey experiments:
    plot_config = {1: {"range": (0, 100), "axvline": []}}
else:
    gen_train = nps.construct_predefined_gens(
        torch.float32,
        seed=10,
        batch_size=args.batch_size,
        num_tasks=2**14,
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        pred_logpdf=False,
        pred_logpdf_diag=False,
        device=device,
    )[args.data]
    gen_cv = nps.construct_predefined_gens(
        torch.float32,
        seed=20,  # Use a different seed!
        batch_size=args.batch_size,
        num_tasks=2**12,  # Lower the number of tasks.
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        pred_logpdf=True,
        pred_logpdf_diag=True,
        device=device,
    )[args.data]

    def gens_eval():
        return [
            (
                name,
                nps.construct_predefined_gens(
                    torch.float32,
                    seed=30,  # Use yet another seed!
                    batch_size=args.batch_size,
                    # Use a high number of tasks.
                    num_tasks=2**6 if args.evaluate_fast else 2**14,
                    dim_x=args.dim_x,
                    dim_y=args.dim_y,
                    pred_logpdf=True,
                    pred_logpdf_diag=True,
                    device=device,
                    x_range_context=x_range_context,
                    x_range_target=x_range_target,
                )[args.data],
            )
            for name, x_range_context, x_range_target in [
                ("interpolation in training range", (-2, 2), (-2, 2)),
                ("interpolation beyond training range", (2, 6), (2, 6)),
                ("extrapolation beyond training range", (-2, 2), (2, 4)),
            ]
        ]

    # Architecture choices specific for the GP{ experiments:
    dws_receptive_field = 4
    margin = 0.1
    if args.dim_x == 1:
        points_per_unit = 64
    elif args.dim_x == 2:
        # Reduce the PPU to reduce memory consumption.
        points_per_unit = 32
        # Since the PPU is reduced, we can also take off a layer of the UNet.
        unet_channels = unet_channels[:-1]
    else:
        raise RuntimeError(f"Invalid input dimensionality {args.dim_x}.")

    # Other settings specific to the GP experiments:
    plot_config = {
        1: {"range": (-2, 4), "axvline": [2]},
        2: {"range": (-2, 2)},
    }


# Construct the model.
if args.model == "cnp":
    model = nps.construct_gnp(
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        dim_embedding=dim_embedding,
        num_enc_layers=num_layers,
        num_dec_layers=num_layers,
        width=width,
        likelihood="het",
    )
elif args.model == "gnp":
    model = nps.construct_gnp(
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        dim_embedding=dim_embedding,
        num_enc_layers=num_layers,
        num_dec_layers=num_layers,
        width=width,
        likelihood="lowrank",
        num_basis_functions=num_basis_functions,
    )
elif args.model == "np":
    model = nps.construct_gnp(
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        dim_embedding=dim_embedding,
        num_enc_layers=num_layers,
        num_dec_layers=num_layers,
        width=width,
        likelihood="het",
        dim_lv=dim_embedding,
    )
elif args.model == "acnp":
    model = nps.construct_agnp(
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        dim_embedding=dim_embedding,
        num_heads=num_heads,
        num_enc_layers=num_layers,
        num_dec_layers=num_layers,
        width=width,
        likelihood="het",
    )
elif args.model == "agnp":
    model = nps.construct_agnp(
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        dim_embedding=dim_embedding,
        num_heads=num_heads,
        num_enc_layers=num_layers,
        num_dec_layers=num_layers,
        width=width,
        likelihood="lowrank",
        num_basis_functions=num_basis_functions,
    )
elif args.model == "anp":
    model = nps.construct_agnp(
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        dim_embedding=dim_embedding,
        num_heads=num_heads,
        num_enc_layers=num_layers,
        num_dec_layers=num_layers,
        width=width,
        likelihood="het",
        dim_lv=dim_embedding,
    )
elif args.model == "convcnp":
    model = nps.construct_convgnp(
        points_per_unit=points_per_unit,
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        likelihood="het",
        conv_arch=args.arch,
        unet_channels=unet_channels,
        dws_channels=dws_channels,
        dws_layers=num_layers,
        dws_receptive_field=dws_receptive_field,
        margin=margin,
    )
elif args.model == "convgnp":
    model = nps.construct_convgnp(
        points_per_unit=points_per_unit,
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        likelihood="lowrank",
        conv_arch=args.arch,
        unet_channels=unet_channels,
        dws_channels=dws_channels,
        dws_layers=num_layers,
        dws_receptive_field=dws_receptive_field,
        num_basis_functions=num_basis_functions,
        margin=margin,
    )
elif args.model == "convnp":
    model = nps.construct_convgnp(
        points_per_unit=points_per_unit,
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        likelihood="het",
        conv_arch=args.arch,
        unet_channels=unet_channels,
        dws_channels=dws_channels,
        dws_layers=num_layers,
        dws_receptive_field=dws_receptive_field,
        dim_lv=16,
        margin=margin,
    )
elif args.model == "fullconvgnp":
    model = nps.construct_fullconvgnp(
        points_per_unit=points_per_unit,
        dim_x=args.dim_x,
        dim_y=args.dim_y,
        conv_arch=args.arch,
        unet_channels=unet_channels,
        dws_channels=dws_channels,
        dws_layers=num_layers,
        dws_receptive_field=dws_receptive_field,
        margin=margin,
    )
else:
    raise ValueError(f'Invalid model "{args.model}".')

# Ensure that the model is on the GPU and print some statistics.
model = model.to(device)
out.kv("Number of parameters", nps.num_params(model))

# Setup training objective.
if args.objective == "loglik":
    objective = partial(
        nps.loglik,
        num_samples=args.num_samples,
        normalise=True,
    )
    objective_cv = partial(
        nps.loglik,
        num_samples=args.num_samples,
        normalise=True,
    )
    objectives_eval = [
        (
            "Loglik",
            partial(
                nps.loglik,
                num_samples=args.evaluate_num_samples,
                batch_size=args.evaluate_batch_size,
                normalise=True,
            ),
        )
    ]
elif args.objective == "elbo":
    objective = partial(
        nps.elbo,
        num_samples=args.num_samples,
        subsume_context=True,
        normalise=True,
    )
    objective_cv = partial(
        nps.elbo,
        num_samples=args.num_samples,
        subsume_context=False,  # Lower bound the right quantity.
        normalise=True,
    )
    objectives_eval = [
        (
            "ELBO",
            partial(
                nps.elbo,
                # Don't need a high number of samples, because it is unbiased.
                num_samples=5,
                subsume_context=False,  # Lower bound the right quantity.
                normalise=True,
            ),
        ),
        (
            "Loglik",
            partial(
                nps.loglik,
                num_samples=args.evaluate_num_samples,
                batch_size=args.evaluate_batch_size,
                normalise=True,
            ),
        ),
    ]
else:
    raise RuntimeError(f'Invalid objective "{args.objective}".')

# The user can just want to see some statistics about the model.
if args.no_action:
    exit()

if args.evaluate:
    # Perform evaluation.
    if args.evaluate_last:
        name = "model-last.torch"
    else:
        name = "model-best.torch"
    model.load_state_dict(torch.load(wd.file(name), map_location=device))

    if not args.ar:
        # Make some plots.
        for i in range(args.evaluate_plot_num_samples):
            visualise(
                model,
                gen_cv,
                name="evaluate",
                epoch=i + 1,
                config=plot_config,
            )

        # For every objective and evaluation generator, do the evaluation.
        for objecive_name, objective_eval in objectives_eval:
            with out.Section(objecive_name.capitalize()):
                for gen_name, gen in gens_eval():
                    with out.Section(gen_name.capitalize()):
                        state, _ = eval(state, model, objective_eval, gen)

    # Do AR evaluation, but only for the conditional models.
    # TODO: Enable this for all input and output dimensionalities once possible.
    if args.model in {"cnp", "acnp", "convcnp"} and args.dim_x == args.dim_y == 1:
        # Make some plots.
        for i in range(args.evaluate_plot_num_samples):
            visualise(
                model,
                gen_cv,
                name="evaluate-ar",
                epoch=i + 1,
                predict=nps.ar_predict,
                config=plot_config,
            )

        # For both random and left-to-right ordering, do AR testing.
        for order in ["random", "left-to-right"]:
            with out.Section(order.capitalize()):
                for name, gen in gens_eval():
                    with out.Section(name.capitalize()):
                        state, _ = eval(
                            state,
                            model,
                            partial(nps.ar_loglik, order=order, normalise=True),
                            gen,
                        )
else:
    # Perform training. First, check if we want to resume training.
    start = 0
    if args.resume_at_epoch:
        start = args.resume_at_epoch - 1
        model.load_state_dict(
            torch.load(wd.file("model-last.torch"), map_location=device)
        )

    # Setup training loop.
    opt = torch.optim.Adam(model.parameters(), args.rate)
    best_eval_lik = -np.inf

    for i in range(start, args.epochs):
        with out.Section(f"Epoch {i + 1}"):
            # Perform an epoch.
            state = train(state, model, objective, gen_train, epoch=i)

            # Save current model.
            torch.save(model.state_dict(), wd.file(f"model-last.torch"))

            # The epoch is done. Now evaluate.
            state, val = eval(state, model, objective_cv, gen_cv)

            # Check if the model is the new best. If so, save it.
            if val > best_eval_lik:
                out.out("New best model!")
                best_eval_lik = val
                torch.save(model.state_dict(), wd.file(f"model-best.torch"))

            # Visualise a prediction by the model.
            visualise(
                model,
                gen_cv,
                name="train-epoch",
                epoch=i + 1,
                config=plot_config,
            )
