from enum import Enum


class Groups(Enum):
    MARGINAL_DENSITIES = "marginal_densities"
    TRAJECTORIES = "trajectories"


class Datasets(Enum):
    LOG_LIKELIHOODS = "log_likelihoods"
    MEANS = "means"
    VARIANCES = "variances"
