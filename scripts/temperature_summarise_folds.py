import lab as B
import wbml.out as out
import torch
from wbml.experiment import WorkingDirectory
import neuralprocesses.torch as nps

import experiment
from train import main

wd = WorkingDirectory("_experiments", "temperature_summarise_folds")


def compute_logpdfs_maes(data, model):
    # Load experiment.
    exp = main(data=data, model=model, load=True)

    # Setup model.
    model = exp["model"]
    model.load_state_dict(torch.load(exp["wd"].file("model-best.torch"))["weights"])

    # Setup generator.
    gens = exp["gens_eval"]()
    _, gen = gens[0]  # The first one corresponds to downscaling.

    state = B.create_random_state(torch.float32, seed=0)
    logpdfs, maes = [], []
    with torch.no_grad():
        for batch in gen.epoch():
            state, pred = model(state, batch["contexts"], batch["xt"])
            n = nps.num_data(batch["xt"], batch["yt"])
            logpdfs.append(pred.logpdf(batch["yt"]) / n)
            maes.append(B.abs(pred.mean - batch["yt"]))
    return B.concat(*logpdfs), B.concat(*maes)


for model in ["convcnp-mlp", "convgnp-mlp"]:
    with out.Section(model):
        for data in ["temperature-germany", "temperature-value"]:
            with out.Section(data):
                logpdfs, maes = [], []
                for fold in [1, 2, 3, 4, 5]:
                    fold_logpdfs, fold_maes = compute_logpdfs_maes(
                        f"{data}-{fold}",
                        model,
                    )
                    maes.append(fold_maes)
                    logpdfs.append(fold_logpdfs)
                logpdfs = B.concat(*logpdfs)
                maes = B.concat(*maes)

                # Compute the average MAE per station, and then take the median over
                # stations. This lines up with the VALUE protocol.
                maes = B.nanmean(maes, axis=(0, 1))

                out.kv("Loglik", experiment.with_err(logpdfs, and_upper=True))
                out.kv("MAE", experiment.with_err(maes))
                out.kv("MAE (median)", experiment.with_err(*experiment.median_and_err(maes)))
