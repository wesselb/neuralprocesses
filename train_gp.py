import argparse

import lab as B
from mlkernels import EQ
from wbml.out import Progress

from neuralprocesses.data import GPGenerator


parser = argparse.ArgumentParser()
parser.add_argument("--dim_x", type=int, default=1)
parser.add_argument("--dim_y", type=int, default=1)
parser.add_argument("--backend", choices=["tensorflow", "torch"], required=True)
parser.add_argument("--batch_size", type=int, default=128)
parser.add_argument("--harmonics", type=int, default=0)
args = parser.parse_args()

batch_size = args.batch_size
dim_x = args.dim_x
dim_y = args.dim_y
num_harmonics = args.harmonics


if dim_x == 1:
    kernel = EQ().stretch(1) * EQ().periodic(0.25)
    unet_channels = (256,) * 6
    B.epsilon = 1e-7
    rate = 1e-4
    points_per_unit = 64
elif dim_x == 2:
    kernel = EQ().stretch(1) * EQ().periodic(1)
    unet_channels = (128,) * 6
    B.epsilon = 1e-7
    rate = 5e-5
    points_per_unit = 20
else:
    raise RuntimeError("Could not determine kernel for input dimensionality.")

if args.backend == "torch":

    import torch as backend
    import neuralprocesses.torch as nps

    if backend.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"

    def to_device(x):
        return x.to(device)

    def create_optimiser(model):
        return backend.optim.Adam(model.parameters(), rate)

    def step_optimiser(opt, model, f):
        vals = f()
        val = B.mean(vals)
        opt.zero_grad(set_to_none=True)
        val.backward()
        opt.step()
        return vals

elif args.backend == "tensorflow":

    import tensorflow as backend
    import neuralprocesses.tensorflow as nps

    if len(backend.config.list_physical_devices("GPU")) > 0:
        device = "gpu"
    else:
        device = "cpu"

    def to_device(x):
        return x

    def create_optimiser(model):
        return backend.keras.optimizers.Adam(rate)

    def step_optimiser(opt, model, f):
        with backend.GradientTape() as tape:
            vals = f()
            val = B.mean(vals)
        grads = tape.gradient(val, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))
        return vals

else:
    raise ValueError(f'Unknown backend "{args.backend}".')


B.set_global_device(device)

# model = to_device(
#     nps.construct_convgnp(
#         points_per_unit=points_per_unit,
#         dim_x=dim_x,
#         dim_y=dim_y,
#         likelihood="lowrank",
#         harmonics_range=(-2, 2),
#         num_harmonics=num_harmonics,
#         unet_channels=unet_channels,
#         num_basis_functions=1024,
#     )
# )

# Discretisation of the functional embedding:
disc = nps.Discretisation(
    points_per_unit=64,
    multiple=1,
    margin=0.1,
    dim=dim_x,
)
# DWS CNN:
cnn = nps.ConvNet(
    dim=dim_x,
    in_channels=2 * dim_y,
    out_channels=(2 + 1024) * dim_y,
    num_layers=6,
    channels=64,
    receptive_field=4,
    points_per_unit=disc.points_per_unit,
)

# Create the encoder and decoder and construct the model.
encoder = nps.FunctionalCoder(
    disc,
    nps.Chain(
        nps.PrependDensityChannel(),
        nps.SetConv(disc.points_per_unit),
        nps.DivideByFirstChannel(),
    ),
)
decoder = nps.Chain(
    cnn,
    nps.SetConv(disc.points_per_unit),
    nps.LowRankGaussianLikelihood(1024),
)
model = to_device(nps.Model(encoder, decoder))


gen = GPGenerator(
    backend.float32,
    kernel=kernel,
    batch_size=batch_size,
    num_context_points=(3, 50),
    num_target_points=50,
    x_ranges=((-2, 2),) * dim_x,
    dim_y=dim_y,
    pred_logpdf=True,
    pred_logpdf_diag=True,
    device=device,
)
gen_eval = GPGenerator(
    backend.float32,
    kernel=kernel,
    num_tasks=2 ** 10,
    batch_size=16,
    num_context_points=(3, 50),
    num_target_points=50,
    x_ranges=((-2, 2),) * dim_x,
    dim_y=dim_y,
    pred_logpdf=True,
    pred_logpdf_diag=True,
    device=device,
)


def objective(xc, yc, xt, yt):
    pred = model(xc, yc, xt)
    # Use `float64`s for the logpdf computation.
    pred = B.cast(backend.float64, pred)
    return -pred.logpdf(B.cast(backend.float64, yt))


def with_err(vals):
    vals = B.to_numpy(vals)
    mean = B.mean(vals)
    err = 1.96 * B.std(vals) / B.sqrt(B.length(vals))
    return f"{mean:7.3f} +- {err:7.3f}"


opt = create_optimiser(model)

with Progress(name="Epochs", total=10_000, filter_global=None) as progress_epochs:
    for i in range(10_000):
        with Progress(name=f"Epoch {i + 1}", total=gen.num_batches) as progress_epoch:
            for batch in gen.epoch():
                vals = step_optimiser(
                    opt,
                    model,
                    lambda: objective(
                        batch["xc"],
                        batch["yc"],
                        batch["xt"],
                        batch["yt"],
                    ),
                )
                nt = B.shape(batch["xt"], 2)
                progress_epoch(
                    {
                        "KL (full)": with_err((vals + batch["pred_logpdf"]) / nt),
                        "KL (diag)": with_err((vals + batch["pred_logpdf_diag"]) / nt),
                        "Score": B.mean(vals + batch["pred_logpdf"])
                        / B.mean(-batch["pred_logpdf_diag"] + batch["pred_logpdf"]),
                    }
                )

        with backend.no_grad():
            full_vals = []
            diag_vals = []
            for batch in gen_eval.epoch():
                batch_vals = objective(
                    batch["xc"],
                    batch["yc"],
                    batch["xt"],
                    batch["yt"],
                )
                nt = B.shape(batch["xt"], 2)
                full_vals.append(B.to_numpy(batch_vals + batch["pred_logpdf"]) / nt)
                diag_vals.append(
                    B.to_numpy(batch_vals + batch["pred_logpdf_diag"]) / nt
                )
            full_vals = B.concat(*full_vals)
            diag_vals = B.concat(*diag_vals)
            progress_epochs(
                {
                    "KL (full)": with_err(full_vals),
                    "KL (diag)": with_err(diag_vals),
                    "Score": B.mean(full_vals) / B.mean(full_vals - diag_vals),
                }
            )
