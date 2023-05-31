import neuralprocesses as nps  # This fixes inspection below.

__all__ = [
    "construct_likelihood",
    "parse_transform",
]


def construct_likelihood(nps=nps, *, spec, dim_y, num_basis_functions, dtype):
    """Construct the likelihood.

    Args:
        nps (module): Appropriate backend-specific module.
        spec (str, optional): Specification. Must be one of `"het"`, `"lowrank"`,
            `"dense"`, or `"spikes-beta"`. Defaults to `"lowrank"`. Must be given as
            a keyword argument.
        dim_y (int): Dimensionality of the outputs. Must be given as a keyword argument.
        num_basis_functions (int): Number of basis functions for the low-rank
            likelihood. Must be given as a keyword argument.
        dtype (dtype): Data type. Must be given as a keyword argument.

    Returns:
        int: Number of channels that the likelihood requires.
        coder: Coder which can select a particular output channel. This coder may be
            `None`.
        coder: Coder.
    """
    if spec == "het":
        num_channels = 2 * dim_y
        selector = nps.SelectFromChannels(dim_y, dim_y)
        lik = nps.HeterogeneousGaussianLikelihood()
    elif spec == "lowrank":
        num_channels = (2 + num_basis_functions) * dim_y
        selector = nps.SelectFromChannels(dim_y, (num_basis_functions, dim_y), dim_y)
        lik = nps.LowRankGaussianLikelihood(num_basis_functions)
    elif spec == "dense":
        # This is intended to only work for global variables.
        num_channels = 2 * dim_y + dim_y * dim_y
        selector = None
        lik = nps.Chain(
            nps.Splitter(2 * dim_y, dim_y * dim_y),
            nps.Parallel(
                lambda x: x,
                nps.Chain(
                    nps.ToDenseCovariance(),
                    nps.DenseCovariancePSDTransform(),
                ),
            ),
            nps.DenseGaussianLikelihood(),
        )
    elif spec == "spikes-beta":
        num_channels = (2 + 3) * dim_y  # Alpha, beta, and three log-probabilities
        selector = nps.SelectFromChannels(dim_y, dim_y, dim_y, dim_y, dim_y)
        lik = nps.SpikesBetaLikelihood()

    else:
        raise ValueError(f'Incorrect likelihood specification "{spec}".')
    return num_channels, selector, lik


def parse_transform(nps=nps, *, transform):
    """Construct the likelihood.

    Args:
        nps (module): Appropriate backend-specific module.
        transform (str or tuple[float, float]): Bijection applied to the
            output of the model. This can help deal with positive of bounded data.
            Must be either `"positive"`, `"exp"`, `"softplus"`, or
            `"softplus_of_square"` for positive data or `(lower, upper)` for data in
            this open interval.

    Returns:
        coder: Transform.
    """
    if isinstance(transform, str) and transform.lower() in {"positive", "exp"}:
        transform = nps.Transform.exp()
    elif isinstance(transform, str) and transform.lower() == "softplus":
        transform = nps.Transform.softplus()
    elif isinstance(transform, str) and transform.lower() == "softplus_of_square":
        transform = nps.Chain(
            nps.Transform.signed_square(),
            nps.Transform.softplus(),
        )
    elif isinstance(transform, tuple):
        lower, upper = transform
        transform = nps.Transform.bounded(lower, upper)
    elif transform is not None:
        raise ValueError(f'Cannot parse value "{transform}" for `transform`.')
    else:
        transform = lambda x: x
    return transform
