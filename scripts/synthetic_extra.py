import lab.torch as B
import torch
import wbml.out as out
from wbml.experiment import WorkingDirectory

import neuralprocesses.torch as nps
from experiment import with_err

wd = WorkingDirectory("_experiments", "synthetic_extra")


def gens_eval(data, dim_x, dim_y):
    return [
        (
            eval_name,
            nps.construct_predefined_gens(
                torch.float32,
                seed=30,
                batch_size=16,
                num_tasks=2**12,
                dim_x=dim_x,
                dim_y=dim_y,
                pred_logpdf=True,
                pred_logpdf_diag=True,
                device="cuda",
                x_range_context=x_range_context,
                x_range_target=x_range_target,
            )[data],
        )
        for eval_name, x_range_context, x_range_target in [
            ("interpolation in training range", (-2, 2), (-2, 2)),
            ("interpolation beyond training range", (2, 6), (2, 6)),
            ("extrapolation beyond training range", (-2, 2), (2, 6)),
        ]
    ]


for data in ["eq", "matern", "weakly-periodic", "sawtooth", "mixture"]:
    for dim_x in [1, 2]:
        for dim_y in [1, 2]:
            with out.Section(f"{data}-{dim_x}-{dim_y}"):
                for task, gen in gens_eval(data, dim_x, dim_y):
                    with out.Section(task.capitalize()):

                        logpdfs = []
                        logpdfs_diag = []
                        m1s = [0] * dim_y
                        m2s = [0] * dim_y
                        ns = [0] * dim_y

                        # Loop over the epoch and compute statistics.
                        for batch in gen.epoch():
                            if "pred_logpdf" in batch:
                                logpdfs.append(
                                    batch["pred_logpdf"]
                                    / nps.num_data(batch["xt"], batch["yt"])
                                )
                                logpdfs_diag.append(
                                    batch["pred_logpdf_diag"]
                                    / nps.num_data(batch["xt"], batch["yt"])
                                )
                            if dim_y == 1:
                                m1s[0] += B.sum(batch["yt"])
                                m2s[0] += B.sum(batch["yt"] ** 2)
                                ns[0] += B.length(batch["yt"])
                            else:
                                for i in range(dim_y):
                                    m1s[i] += B.sum(batch["yt"][i])
                                    m2s[i] += B.sum(batch["yt"][i] ** 2)
                                    ns[i] += B.length(batch["yt"][i])

                        # Compute the trivial logpdf.
                        logpdfs_trivial = []
                        for i in range(dim_y):
                            m1 = m1s[i] / ns[i]
                            m2 = m2s[i] / ns[i]
                            emp_var = m2 - m1**2
                            logpdfs_trivial.append(
                                -0.5 * B.log(2 * B.pi * B.exp(1) * emp_var)
                            )
                        logpdf_trivial = B.mean(B.stack(*logpdfs_trivial))
                        out.kv("Logpdf (trivial)", logpdf_trivial, fmt=".5f")

                        # Report  KLs.
                        if logpdfs:
                            out.kv("Logpdf (diag)", with_err(B.stack(*logpdfs_diag)))
                            out.kv(
                                "KL (diag)",
                                with_err(B.stack(*logpdfs) - B.stack(*logpdfs_diag)),
                            )
                            out.kv(
                                "KL (trivial)",
                                with_err(B.stack(*logpdfs) - logpdf_trivial),
                            )
