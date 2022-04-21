import lab as B

import neuralprocesses as nps  # This fixes inspection below.

__all__ = ["construct_likelihood"]


def construct_likelihood(nps=nps, *, spec, dim_y, num_basis_functions, dtype):
    """Construct the likelihood.

    Args:
        nps (module): Appropriate backend-specific module.
        spec (str, optional): Specification. Must be one of `"het"`, `"lowrank"`, or
            `"dense"`. Defaults to `"lowrank"`. Must be given as a keyword argument.
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
    elif spec == "dense":
        # This will only work for global variables!
        num_channels = 2 * dim_y + dim_y * dim_y
        lik = nps.Chain(
            nps.Splitter(2 * dim_y, dim_y * dim_y),
            nps.Parallel(
                # The split for the mean is alright.
                lambda x: x,
                nps.Chain(
                    # For the variance, first make it positive definite. Assume that
                    # `n = 1`, so we can ignore the data dimension.
                    lambda x: B.reshape(x, *B.shape(x)[:-2], dim_y, dim_y),
                    # Make PD and divide by 100 to stabilise initialisation.
                    lambda x: B.matmul(x, x, tr_b=True) / 100,
                    # Make it of the shape `(*b, c, n, c, n)` with `n = 1`.
                    lambda x: x[..., :, None, :, None],
                ),
            ),
            nps.DenseGaussianLikelihood(),
        )

    else:
        raise ValueError(f'Incorrect likelihood specification "{spec}".')
    return num_channels, lik
