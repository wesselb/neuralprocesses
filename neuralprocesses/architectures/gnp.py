from .lik import construct_likelihood
from ..util import register_model

__all__ = ["create_construct_gnp"]


@register_model
def create_construct_gnp(ns):
    def construct_gnp(
        dim_x=1,
        dim_y=1,
        dim_embedding=64,
        num_enc_layers=6,
        num_dec_layers=6,
        likelihood="het",
        num_basis_functions=64,
        dtype=None,
    ):
        mlp_out_channels, likelihood = construct_likelihood(
            ns,
            spec=likelihood,
            dim_y=dim_y,
            num_basis_functions=num_basis_functions,
        )
        encoder = ns.Chain(
            ns.Parallel(
                ns.InputsCoder(),
                ns.DeepSet(
                    ns.MLP(
                        dim_in=dim_x + dim_y,
                        dim_hidden=dim_embedding,
                        dim_out=dim_embedding,
                        num_layers=num_enc_layers,
                        dtype=dtype,
                    ),
                    ns.MLP(
                        dim_in=dim_embedding,
                        dim_hidden=dim_embedding,
                        dim_out=dim_embedding,
                        num_layers=num_enc_layers,
                        dtype=dtype,
                    ),
                ),
            ),
        )
        decoder = ns.Chain(
            ns.Materialise(),
            ns.MLP(
                dim_in=dim_x + dim_embedding,
                dim_hidden=dim_embedding,
                dim_out=mlp_out_channels,
                num_layers=num_dec_layers,
                dtype=dtype,
            ),
            likelihood,
        )
        return ns.Model(encoder, decoder)

    return construct_gnp
