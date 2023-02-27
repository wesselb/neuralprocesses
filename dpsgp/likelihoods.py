import math

import torch
from torch import nn


class GaussianLikelihood(nn.Module):
    def __init__(self, noise: float):
        super().__init__()

        self.log_noise = nn.Parameter(torch.as_tensor(noise).log())

    @property
    def noise(self):
        return self.log_noise.exp()

    @noise.setter
    def noise(self, value: float):
        self.log_noise = nn.Parameter(torch.as_tensor(value).log())

    def expected_log_prob(
        self, target: torch.Tensor, pred_dist: torch.distributions.MultivariateNormal
    ) -> torch.Tensor:
        mean, variance = pred_dist.mean, pred_dist.variance
        return -0.5 * (
            ((target - mean) ** 2 + variance) / self.noise
            + self.noise.log()
            + math.log(2 * math.pi)
        )

    def forward(self, out: torch.Tensor) -> torch.distributions.Normal:
        return torch.distributions.Normal(out, self.noise.pow(0.5))
