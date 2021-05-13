import lab.torch as B
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

for batch in gen.epoch(device):
    mean, cov = model(batch["x_context"], batch["y_context"], batch["x_target"])
    dist = torch.distributions.MultivariateNormal(loc=mean, covariance_matrix=cov)
    loss = -B.mean(dist.log_prob(batch["y_target"])) / B.shape(batch["y_target"])[1]
    print(loss)

    loss.backward()
    opt.step()
    opt.zero_grad()









