from .lik import construct_likelihood
from ..util import register_model

__all__ = ["construct_gnp"]


@register_model
def construct_gnp(nps):
    def construct_gnp(
        dim_x=1,
        dim_y=1,
        dim_embedding=128,
        num_enc_layers=6,
        num_dec_layers=6,
        likelihood="lowrank",
        num_basis_functions=512,
        dtype=None,
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
                        dim_in=dim_x + dim_y,
                        dim_hidden=512,
                        dim_out=dim_embedding,
                        num_layers=num_enc_layers // 2,
                        dtype=dtype,
                    ),
                    nps.MLP(
                        dim_in=dim_embedding,
                        dim_hidden=512,
                        dim_out=dim_embedding,
                        num_layers=num_enc_layers // 2,
                        dtype=dtype,
                    ),
                ),
            ),
        )
        decoder = nps.Chain(
            nps.Materialise(),
            nps.MLP(
                dim_in=dim_x + dim_embedding,
                dim_hidden=512,
                dim_out=mlp_out_channels,
                num_layers=num_dec_layers,
                dtype=dtype,
            ),
            likelihood,
        )
        return nps.Model(encoder, decoder)

    return construct_gnp
