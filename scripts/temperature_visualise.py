import sys

import lab as B
import torch
import wbml.out as out

from experiment.util import with_err
from train import main
import neuralprocesses.torch as nps

# Load experiment.
exp = main(
    data="temperature",
    model="convgnp-multires",
    root="aws_run_2022-05-17_temperature/_experiments",
    load=True,
)
model = exp["model"]
model.load_state_dict(torch.load(exp["wd"].file("model-last.torch"))["weights"])
gen = nps.TemperatureGenerator(torch.float32, subset="train")
b = gen.generate_batch()

mean, var, noiseless_samples, noisy_samples = nps.ar_predict(
    model,
    b["contexts"],
    nps.AggregateInput((b["xt"], 0)),
)
