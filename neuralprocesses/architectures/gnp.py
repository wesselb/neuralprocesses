import neuralprocesses as nps  # This fixes inspection below.
from .lik import construct_likelihood
from ..util import register_model

__all__ = ["construct_gnp"]


@register_model
def construct_gnp(
    dim_x=1,
    dim_y=1,
    dim_embedding=128,
    num_enc_layers=6,
    num_dec_layers=6,
    likelihood="lowrank",
    num_basis_functions=512,
    dtype=None,
    nps=nps,
):
    mlp_out_channels, likelihood = construct_likelihood(
        nps,
        spec=likelihood,
        dim_y=dim_y,
        num_basis_functions=num_basis_functions,
        dtype=dtype,
    )
    encoder = nps.Chain(
        nps.Parallel(
            nps.InputsCoder(),
            nps.DeepSet(
                nps.MLP(
                    in_dim=dim_x + dim_y,
                    dims=(512,) * (num_enc_layers // 2),
                    out_dim=dim_embedding,
                    dtype=dtype,
                ),
                nps.MLP(
                    in_dim=dim_embedding,
                    dims=(512,) * (num_enc_layers - num_enc_layers // 2),
                    out_dim=dim_embedding,
                    dtype=dtype,
                ),
            ),
        ),
    )
    decoder = nps.Chain(
        nps.Materialise(),
        nps.MLP(
            in_dim=dim_x + dim_embedding,
            dims=(512,) * num_dec_layers,
            out_dim=mlp_out_channels,
            dtype=dtype,
        ),
        likelihood,
    )
    return nps.Model(encoder, decoder)
