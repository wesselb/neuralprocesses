import wbml.out as out
import torch
import lab.torch as B
from wbml.experiment import WorkingDirectory

import neuralprocesses.torch as nps
from experiment import with_err

wd = WorkingDirectory("_experiments", "synthetic_diag_gp")


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


for data in ["eq", "matern", "weakly-periodic"]:
    for dim_x in [1, 2]:
        for dim_y in [1, 2]:
            with out.Section(f"{data}-{dim_x}-{dim_y}"):
                for task, gen in gens_eval(data, dim_x, dim_y):
                    with out.Section(task.capitalize()):
                        out.kv(
                            "Diag",
                            with_err(
                                B.stack(
                                    *[
                                        (
                                            batch["pred_logpdf"]
                                            - batch["pred_logpdf_diag"]
                                        )
                                        / nps.num_data(batch["xt"], batch["yt"])
                                        for batch in gen.epoch()
                                    ]
                                )
                            ),
                        )
