__all__ = ["construct_likelihood"]


def construct_likelihood(ns, *, spec, dim_y, num_basis_functions):
    if spec == "het":
        num_channels = 2 * dim_y
        lik = ns.HeterogeneousGaussianLikelihood()
    elif spec == "lowrank":
        num_channels = (2 + num_basis_functions) * dim_y
        lik = ns.LowRankGaussianLikelihood(num_basis_functions)
    else:
        raise ValueError(f'Incorrect likelihood specification "{likelihood}".')
    return num_channels, lik
