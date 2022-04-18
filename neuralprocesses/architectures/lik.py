import lab as B

import neuralprocesses as nps  # This fixes inspection below.

__all__ = ["construct_likelihood"]


def construct_likelihood(nps=nps, *, spec, dim_y, num_basis_functions, dtype):
    """Construct the likelihood.

    Args:
        nps (module): Appropriate backend-specific module.
        spec (str, optional): Specification. Must be one of "het" or "lowrank".
            Defaults to "lowrank". Must be given as a keyword argument.
        dim_y (int): Dimensionality of the outputs. Must be given as a keyword argument.
        num_basis_functions (int): Number of basis functions for the low-rank
            likelihood. Must be given as a keyword argument.
        dtype (dtype): Data type. Must be given as a keyword argument.

    Returns:
        tuple[int, coder]: Number of channels that the likelihood requires and the
            likelihood.
    """
    if spec == "het":
        num_channels = 2 * dim_y
        lik = nps.HeterogeneousGaussianLikelihood()
    elif spec == "lowrank":
        num_channels = (2 + num_basis_functions) * dim_y
        lik = nps.LowRankGaussianLikelihood(num_basis_functions)
    else:
        raise ValueError(f'Incorrect likelihood specification "{spec}".')
    return num_channels, lik
