import argparse
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


def train(state, model, objective, gen):
    """Train for an epoch."""
    for batch in gen.epoch():
        state, obj = objective(
            state,
            model,
            batch["xc"],
            batch["yc"],
            batch["xt"],
            batch["yt"],
        )
        # Be sure to negate the output of `objective`.
        val = -B.mean(obj)
        opt.zero_grad(set_to_none=True)
        val.backward()
        opt.step()
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
        out.kv("Loglik", with_err(B.concat(*vals)))
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


def first_np(x):
    """Get the first batch and convert to NumPy."""
    if B.rank(x) == 2:
        return B.to_numpy(x[0, :])
    elif B.rank(x) == 3:
        return B.to_numpy(x[0, 0, :])
    elif B.rank(x) == 4:
        return B.transpose(B.to_numpy(x[:, 0, 0, :]))
    else:
        raise ValueError(f"Rank must be two, three, or four.")


def plot_first_of_batch(model, gen, *, name, epoch):
    """Plot the prediction for the first element of a batch."""
    batch = gen.generate_batch()

    # Define points to predict at.
    with B.on_device(batch["xt"]):
        x = B.linspace(B.dtype(batch["xt"]), -2, 2, 500)[None, None, :]
        x = B.tile(x, B.shape(batch["xt"], 0), 1, 1)

    # Predict with model and produce five noiseless samples.
    with torch.no_grad():
        mean, var, samples = nps.predict(model, batch["xc"], batch["yc"], x)

    plt.figure(figsize=(6, 4))

    # Plot context and target.
    plt.scatter(
        first_np(batch["xc"]),
        first_np(batch["yc"]),
        label="Context",
        style="train",
        s=20,
    )
    plt.scatter(
        first_np(batch["xt"]),
        first_np(batch["yt"]),
        label="Target",
        style="test",
        s=20,
    )

    # Plot prediction.
    err = 1.96 * B.sqrt(var)
    plt.plot(
        first_np(x),
        first_np(mean),
        label="Prediction",
        style="pred",
    )
    plt.fill_between(
        first_np(x),
        first_np(mean - err),
        first_np(mean + err),
        style="pred",
    )
    plt.plot(
        first_np(x),
        first_np(samples),
        style="pred",
        ls="-",
        lw=0.5,
    )

    # Plot prediction by ground truth.
    if hasattr(gen, "kernel"):
        f = stheno.GP(gen.kernel)
        # Make sure that everything is of `float64`s and on the GPU.
        noise = B.to_active_device(B.cast(torch.float64, gen.noise))
        xc = B.transpose(B.cast(torch.float64, batch["xc"]))
        yc = B.transpose(B.cast(torch.float64, batch["yc"]))
        x = B.transpose(B.cast(torch.float64, x))
        # Compute posterior GP.
        f_post = f | (f(xc, noise), yc)
        mean, lower, upper = f_post(x).marginal_credible_bounds()
        plt.plot(
            first_np(B.transpose(x)),
            first_np(mean),
            label="Truth",
            style="pred2",
        )
        plt.plot(
            first_np(B.transpose(x)),
            first_np(lower),
            style="pred2",
        )
        plt.plot(
            first_np(B.transpose(x)),
            first_np(upper),
            style="pred2",
        )

    plt.xlim(B.min(x), B.max(x))
    tweak()
    plt.savefig(wd.file(f"{name}-{epoch:03d}.pdf"))
    plt.close()


# Setup arguments.
parser = argparse.ArgumentParser()
parser.add_argument("--subdir", type=str, nargs="*")
parser.add_argument("--dim-x", type=int, default=1)
parser.add_argument("--dim-y", type=int, default=1)
parser.add_argument("--epochs", type=int, default=200)
parser.add_argument("--rate", type=float, default=1e-4)
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
parser.add_argument("--margin", type=float, default=0.1)
parser.add_argument("--receptive-field", type=float, default=2)
parser.add_argument(
    "--data",
    choices=[
        "eq",
        "matern",
        "weakly-periodic",
        "sawtooth",
        "mixture",
    ],
    default="eq",
)
parser.add_argument("--objective", choices=["loglik", "elbo"], default="loglik")
parser.add_argument("--num-samples", type=int, default=1)
parser.add_argument("--resume-at-epoch", type=int)
parser.add_argument("--evaluate", action="store_true")
parser.add_argument(
    "--evaluate-objective",
    choices=["loglik", "elbo"],
    default="loglik",
)
parser.add_argument("--evaluate-num-samples", type=int, default=4096)
parser.add_argument("--no-action", action="store_true")
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

# Setup script.
out.report_time = True
B.epsilon = 1e-8
wd = WorkingDirectory(
    "_experiments",
    *(args.subdir or ()),
    args.data,
    args.model,
    args.objective,
    *((args.arch,) if args.arch else ()),
    f"x{args.dim_x}_y{args.dim_y}",
    log="log_evaluate.txt" if args.evaluate else "log.txt",
)

# Use a GPU if one is available.
if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"
B.set_global_device(device)
# Maintain an explicit random state through the execution.
state = B.create_random_state(torch.float32, seed=0)

# Setup data generators for training and for evaluation.
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
gens_eval = [
    (
        name,
        nps.construct_predefined_gens(
            torch.float32,
            seed=20,  # Use yet another seed!
            batch_size=args.batch_size,
            num_tasks=2**14,  # Use a high number of tasks.
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
        ("extrapolation beyond training range", (-2, 2), (2, 6)),
    ]
]

# Setup architectures.
width = 256
dim_embedding = 128
num_heads = 8
num_layers = 6
unet_channels = (64,) * num_layers
dws_channels = 64
dws_receptive_field = args.receptive_field
num_basis_functions = 512
if args.dim_x == 1:
    points_per_unit = 40
elif args.dim_x == 2:
    # Reduce the PPU to reduce memory consumption.
    points_per_unit = 20
else:
    raise RuntimeError(f"Invalid input dimensionality {args.dim_x}.")

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
        margin=args.margin,
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
        margin=args.margin,
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
        margin=args.margin,
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
        margin=args.margin,
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
elif args.objective == "elbo":
    objective = partial(
        nps.elbo,
        num_samples=args.num_samples,
        normalise=True,
    )
else:
    raise RuntimeError(f'Invalid objective "{args.objective}".')

# Setup evaluation objective.
if args.evaluate_objective == "loglik":
    evaluate_objective = partial(
        nps.loglik,
        num_samples=args.evaluate_num_samples,
        normalise=True,
    )
elif args.objective == "elbo":
    evaluate_objective = partial(
        nps.elbo,
        num_samples=args.evaluate_num_samples,
        normalise=True,
    )
else:
    raise RuntimeError(f'Invalid objective "{args.objective}".')

# The user can just want to see some statistics about the model.
if args.no_action:
    exit()

if args.evaluate:
    # Perform evaluation. First, load the best model.
    model.load_state_dict(torch.load(wd.file("model-best.torch"), map_location=device))

    # Visualise some predictions by the model.
    if args.dim_x == 1 and args.dim_y == 1:
        for i in range(10):
            plot_first_of_batch(model, gen_cv, name="evaluate", epoch=i + 1)

    for name, gen in gens_eval:
        with out.Section(name.capitalize()):
            eval(state, model, evaluate_objective, gen)
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
            state = train(state, model, objective, gen_train)

            # Save current model.
            torch.save(model.state_dict(), wd.file(f"model-last.torch"))

            # The epoch is done. Now evaluate.
            state, val = eval(state, model, objective, gen_cv)

            # Check if the model is the new best. If so, save it.
            if val > best_eval_lik:
                out.out("New best model!")
                best_eval_lik = val
                torch.save(model.state_dict(), wd.file(f"model-best.torch"))

            # Visualise a prediction by the model.
            if args.dim_x == 1 and args.dim_y == 1:
                plot_first_of_batch(model, gen_cv, name="train-epoch", epoch=i + 1)
