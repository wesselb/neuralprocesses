import lab.torch as B
import numpy as np
import torch

from neuralprocesses.data import GPGenerator
from neuralprocesses.gnp import GNP

if torch.cuda.is_available():
    device = "cuda"
else:
    device = "cpu"


gen = GPGenerator()
model = GNP().to(device)

opt = torch.optim.Adam(model.parameters(), lr=5e-4)


class MovingAverage:
    def __init__(self, size=200):
        self.stack = []
        self.size = size

    def record(self, value):
        self.stack.append(B.to_numpy(value))
        while len(self.stack) > self.size:
            self.stack.pop(0)
        print(B.mean(np.array(self.stack)))


ma = MovingAverage()

while True:
    print("New epoch!")
    for batch in gen.epoch(device):
        mean, cov = model(batch["x_context"], batch["y_context"], batch["x_target"])
        dist = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)
        loss = (
            -B.mean(dist.log_prob(batch["y_target"][:, :, 0]))
            / B.shape(batch["y_target"])[1]
        )

        ma.record(loss)

        loss.backward()
        opt.step()
        opt.zero_grad()
