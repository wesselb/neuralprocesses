from ..util import register_model

__all__ = ["create_construct_cnp"]


@register_model
def create_construct_cnp(ns):
    def construct_cnp(
        dim_x=1,
        dim_y=1,
        dim_embedding=64,
        num_enc_layers=6,
        num_dec_layers=6,
        dtype=None,
    ):
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
                dim_out=2 * dim_y,
                num_layers=num_dec_layers,
                dtype=dtype,
            ),
            ns.HeterogeneousGaussianLikelihood(),
        )
        return ns.Model(encoder, decoder)

    return construct_cnp
