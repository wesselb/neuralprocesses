from .. import (
    Chain,
    Parallel,
    InputsCoder,
    DeepSet,
    MLP,
    Materialise,
    HeterogeneousGaussianLikelihood,
    Model,
)

__all__ = ["construct_cnp"]


# noinspection PyTypeChecker
def construct_cnp(
    dim_x=1,
    dim_y=1,
    dim_embedding=64,
    num_enc_layers=6,
    num_dec_layers=6,
):
    encoder = Chain(
        Parallel(
            InputsCoder(),
            DeepSet(
                MLP(
                    dim_in=dim_x + dim_y,
                    dim_hidden=dim_embedding,
                    dim_out=dim_embedding,
                    num_layers=num_enc_layers,
                ),
                MLP(
                    dim_in=dim_embedding,
                    dim_hidden=dim_embedding,
                    dim_out=dim_embedding,
                    num_layers=num_enc_layers,
                ),
            ),
        ),
    )
    decoder = Chain(
        Materialise(),
        MLP(
            dim_in=dim_x + dim_embedding,
            dim_hidden=dim_embedding,
            dim_out=2 * dim_y,
            num_layers=num_dec_layers,
        ),
        HeterogeneousGaussianLikelihood(),
    )
    return Model(encoder, decoder)
