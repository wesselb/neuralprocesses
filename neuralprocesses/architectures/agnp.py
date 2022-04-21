import neuralprocesses as nps  # This fixes inspection below.
from .util import construct_likelihood, parse_transform
from ..util import register_model

__all__ = ["construct_agnp"]


@register_model
def construct_agnp(
    dim_x=1,
    dim_y=1,
    dim_embedding=256,
    num_heads=8,
    num_enc_layers=6,
    num_dec_layers=6,
    width=512,
    likelihood="lowrank",
    num_basis_functions=512,
    dim_lv=0,
    lv_likelihood="het",
    dtype=None,
    transform=None,
    nps=nps,
):
    """An Attentive Gaussian Neural Process.

    Args:
        dim_x (int, optional): Dimensionality of the inputs. Defaults to 1.
        dim_y (int, optional): Dimensionality of the outputs. Defaults to 1.
        dim_embedding (int, optional): Dimensionality of the embedding. Defaults to 256.
        num_heads (int, optional): Number of heads. Defaults to 16.
        num_enc_layers (int, optional): Number of layers in the encoder. Defaults to 6.
        num_dec_layers (int, optional): Number of layers in the decoder. Defaults to 6.
        width (int, optional): Widths of all intermediate MLPs. Defaults to 512.
        likelihood (str, optional): Likelihood. Must be one of `"het"` or `"lowrank"`.
            Defaults to `"lowrank"`.
        num_basis_functions (int, optional): Number of basis functions for the
            low-rank likelihood. Defaults to 512.
        dim_lv (bool, optional): Dimensionality of the latent variable.
        lv_likelihood (str, optional): Likelihood of the latent variable. Must be one of
            `"het"` or `"dense"`. Defaults to `"het"`.
        transform (str or tuple[float, float], optional): Bijection applied to the
            output of the model. This can help deal with positive of bounded data.
            Must be either `"positive"` for positive data or `(lower, upper)` for data
            in this open interval.
        dtype (dtype, optional): Data type.

    Returns:
        :class:`.model.Model`: GNP model.
    """
    likelihood_in_channels, likelihood = construct_likelihood(
        nps,
        spec=likelihood,
        dim_y=dim_y,
        num_basis_functions=num_basis_functions,
        dtype=dtype,
    )

    if dim_lv > 0:
        lv_likelihood_in_channels, lv_likelihood = construct_likelihood(
            nps,
            spec=lv_likelihood,
            dim_y=dim_lv,
            num_basis_functions=num_basis_functions,
            dtype=dtype,
        )
        lv_encoder = nps.Chain(
            nps.DeepSet(
                nps.MLP(
                    in_dim=dim_x + dim_y,
                    layers=(width,) * (num_enc_layers // 2),
                    out_dim=dim_embedding,
                    dtype=dtype,
                ),
                nps.MLP(
                    in_dim=dim_embedding,
                    layers=(width,) * (num_enc_layers - num_enc_layers // 2),
                    out_dim=lv_likelihood_in_channels,
                    dtype=dtype,
                ),
            ),
            lv_likelihood,
        )

    encoder = nps.Chain(
        nps.Parallel(
            nps.Chain(
                nps.InputsCoder(),
                nps.DeterministicLikelihood(),
            ),
            nps.Chain(
                nps.Attention(
                    dim_x=dim_x,
                    dim_y=dim_y,
                    dim_embedding=dim_embedding,
                    num_heads=num_heads,
                    num_enc_layers=num_enc_layers,
                    dtype=dtype,
                ),
                nps.DeterministicLikelihood(),
            ),
            *((lv_encoder,) if dim_lv > 0 else ())
        ),
    )
    decoder = nps.Chain(
        nps.Materialise(),
        nps.MLP(
            in_dim=dim_x + dim_embedding + dim_lv,
            layers=(width,) * num_dec_layers,
            out_dim=likelihood_in_channels,
            dtype=dtype,
        ),
        likelihood,
        parse_transform(nps, transform=transform),
    )
    return nps.Model(encoder, decoder)
