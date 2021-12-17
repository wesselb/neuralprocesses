import lab as B
import torch
from wbml.out import Progress

import neuralprocesses.torch as nps
from neuralprocesses.data import GPGenerator

B.set_global_device("cuda")

dim_x = 2
dim_y = 2

cnp = nps.construct_convgnp(
    points_per_unit=64,
    dim_x=dim_x,
    dim_y=dim_y,
    likelihood="lowrank",
).to("cuda")

gen = GPGenerator(
    torch.float32,
    batch_size=32,
    num_context_points=(1, 50),
    num_target_points=50,
    x_ranges=((-2, 2),) * dim_x,
    dim_y=dim_y,
    pred_logpdf=True,
    pred_logpdf_diag=True,
    device="cuda",
)


def objective(xc, yc, xt, yt):
    pred = cnp(xc, yc, xt)
    # Use `float64`s for the logpdf computation.
    pred = B.cast(torch.float64, pred)
    return -B.mean(pred.logpdf(B.cast(torch.float64, yt)))


opt = torch.optim.Adam(cnp.parameters(), 1e-3)

for i in range(10_000):
    with Progress(name=f"Epoch {i + 1}", total=gen.num_batches) as progress:
        for batch in gen.epoch():
            opt.zero_grad()
            val = objective(batch["xc"], batch["yc"], batch["xt"], batch["yt"])
            val.backward()
            opt.step()
            nt = B.shape(batch["xt"], 2)
            progress(
                {
                    "Objective (full)": (val + B.mean(batch["pred_logpdf"])) / nt,
                    "Objective (diag)": (val + B.mean(batch["pred_logpdf_diag"])) / nt,
                }
            )
