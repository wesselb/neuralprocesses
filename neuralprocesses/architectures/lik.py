import lab as B

__all__ = ["construct_likelihood"]


def construct_likelihood(nps, *, spec, dim_y, num_basis_functions, dtype):
    if spec == "het":
        num_channels = 2 * dim_y
        lik = nps.HeterogeneousGaussianLikelihood()
    elif spec == "lowrank":
        num_channels = (2 + num_basis_functions) * dim_y
        lik = nps.LowRankGaussianLikelihood(num_basis_functions)
    elif spec == "lowrank-correlated":
        factor = 2
        num_channels = (2 + num_basis_functions) * dim_y + factor * num_basis_functions
        lik = nps.Chain(
            nps.Splitter(
                (2 + num_basis_functions) * dim_y,
                factor * num_basis_functions,
            ),
            nps.Parallel(
                nps.MLP(
                    dim_in=(2 + num_basis_functions) * dim_y,
                    dim_hidden=(2 + num_basis_functions) * dim_y,
                    dim_out=(2 + num_basis_functions) * dim_y,
                    num_layers=3,
                    dtype=dtype
                ),
                nps.Chain(
                    # Compute global channels.
                    lambda x: B.mean(x, axis=-1, squeeze=False),
                    nps.MLP(
                        dim_in=factor * num_basis_functions,
                        dim_hidden=factor * num_basis_functions,
                        dim_out=num_basis_functions * num_basis_functions,
                        num_layers=3,
                        dtype=dtype
                    ),
                ),
            ),
            nps.LowRankGaussianLikelihood(num_basis_functions),
        )
    else:
        raise ValueError(f'Incorrect likelihood specification "{spec}".')
    return num_channels, lik
