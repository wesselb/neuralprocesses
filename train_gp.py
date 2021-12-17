import argparse

import lab as B
from mlkernels import EQ
from wbml.out import Progress

from neuralprocesses.data import GPGenerator

parser = argparse.ArgumentParser()
parser.add_argument("--dim_x", type=int, default=1)
parser.add_argument("--dim_y", type=int, default=1)
parser.add_argument("--backend", choices=["tensorflow", "torch"], required=True)
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

batch_size = args.batch_size
rate = 1e-3 * B.sqrt(args.batch_size / 16)
dim_x = args.dim_x
dim_y = args.dim_y

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
        val = f()
        opt.zero_grad(set_to_none=True)
        val.backward()
        opt.step()
        return val


elif args.backend == "tensorflow":

    import tensorflow as backend
    import neuralprocesses.tensorflow as nps

    if len(backend.config.list_physical_devices("gpu")) > 0:
        device = "gpu"
    else:
        device = "cpu"

    def to_device(x):
        return x

    def create_optimiser(model):
        return backend.keras.optimizers.Adam(rate)

    def step_optimiser(opt, model, f):
        with backend.GradientTape() as tape:
            val = f()
        grads = tape.gradient(val, model.trainable_weights)
        opt.apply_gradients(zip(grads, model.trainable_weights))
        return val


else:
    raise ValueError(f'Unknown backend "{args.backend}".')


B.set_global_device(device)

model = to_device(
    nps.construct_convgnp(
        points_per_unit=64,
        dim_x=dim_x,
        dim_y=dim_y,
        likelihood="lowrank",
    )
)

gen = GPGenerator(
    backend.float32,
    kernel=EQ().stretch(0.25 * B.sqrt(2) ** (dim_x - 1)),
    batch_size=batch_size,
    num_context_points=(1, 50),
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
    return -B.mean(pred.logpdf(B.cast(backend.float64, yt)))


opt = create_optimiser(model)

with Progress(name="Epochs", total=10_000) as progress_epochs:
    for i in range(10_000):
        with Progress(name=f"Epoch {i + 1}", total=gen.num_batches) as progress_epoch:
            for batch in gen.epoch():
                val = step_optimiser(
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
                        "KL (full)": (val + B.mean(batch["pred_logpdf"])) / nt,
                        "KL (diag)": (val + B.mean(batch["pred_logpdf_diag"])) / nt,
                    }
                )
        progress_epochs()
