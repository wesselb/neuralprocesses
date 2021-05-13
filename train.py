import lab.torch as B
import numpy as np
import torch
import stheno

from neuralprocesses.data import GPGenerator
from neuralprocesses.gnp import GNP

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


gen = GPGenerator(kernel=stheno.EQ().stretch(0.25) + 0.05 ** 2 * stheno.Delta())
model = GNP().to(device)

opt = torch.optim.Adam(model.parameters(), lr=5e-4)


def gp_loss(batch):
    total = 0
    for i in range(gen.batch_size):
        m = stheno.Measure()
        p = stheno.GP(stheno.EQ().stretch(0.25), measure=m)
        e1 = 0.05 * stheno.GP(stheno.Delta(), measure=m)
        e2 = 0.05 * stheno.GP(stheno.Delta(), measure=m)
        post = m | ((p + e1)(batch["x_context"][i, :, 0]), batch["y_context"][i, :, 0])
        total += post(p + e2)(batch["x_target"][i, :, 0]).logpdf(
            batch["y_target"][i, :, 0]
        )
    return -total / gen.batch_size / gen.max_test_points


while True:
    print("New epoch!")
    for batch in gen.epoch(device):
        mean, cov = model(batch["x_context"], batch["y_context"], batch["x_target"])
        dist = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)
        loss = (
            -B.mean(dist.log_prob(batch["y_target"][:, :, 0]))
            / B.shape(batch["y_target"])[1]
        )
        print(loss - gp_loss(batch))
        loss.backward()
        opt.step()
        opt.zero_grad()
