import lab as B
import torch
import wbml.out as out
from wbml.experiment import WorkingDirectory

import experiment
from train import main

wd = WorkingDirectory("_experiment", "temperature_mae")


def compute_maes(data, model):
    # Load experiment.
    exp = main(data=data, model=model, load=True)

    # Setup model.
    model = exp["model"]
    model.load_state_dict(torch.load(exp["wd"].file("model-best.torch"))["weights"])

    # Setup generator.
    gens = exp["gens_eval"]()
    _, gen = gens[0]  # The first one corresponds to downscaling.

    state = B.create_random_state(torch.float32, seed=0)
    maes = []
    with torch.no_grad():
        for batch in gen.epoch():
            state, pred = model(state, batch["contexts"], batch["xt"])
            mask = ~B.isnan(batch["yt"])
            maes.append(B.abs(pred.mean[mask] - batch["yt"][mask]))
    return B.concat(*maes)


for model in ["convcnp-mlp", "convgnp-mlp"]:
    with out.Section(model):
        for data in ["temperature-value", "temperature-europe"]:
            with out.Section(data):
                maes = []
                for fold in [1, 2, 3, 4, 5]:
                    maes.append(compute_maes(f"{data}-{fold}", model))
                out.kv("MAE", experiment.with_err(B.concat(*maes), and_upper=True))

        with out.Section("temperature-germany-5"):
            maes = compute_maes("temperature-germany-5", model)
            out.kv("MAE", experiment.with_err(B.concat(*maes), and_upper=True))
