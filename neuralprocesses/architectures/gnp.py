import lab as B
import neuralprocesses as nps  # This fixes inspection below.
from plum import convert

from .util import construct_likelihood, parse_transform
from ..util import register_model

__all__ = ["construct_gnp"]


@register_model
def construct_gnp(
    dim_x=1,
    dim_y=1,
    dim_yc=None,
    dim_yt=None,
    dim_embedding=256,
    attention=False,
    attention_num_heads=8,
    num_enc_layers=3,
    num_dec_layers=6,
    width=512,
    likelihood="lowrank",
    num_basis_functions=512,
    dim_lv=0,
    lv_likelihood="het",
    transform=None,
    dtype=None,
    nps=nps,
):
    """A Gaussian Neural Process.

    Args:
        dim_x (int, optional): Dimensionality of the inputs. Defaults to 1.
        dim_y (int, optional): Dimensionality of the outputs. Defaults to 1.
        dim_yc (int or tuple[int], optional): Dimensionality of the outputs of the
            context set. You should set this if the dimensionality of the outputs
            of the context set is not equal to the dimensionality of the outputs
            of the target set. You should also set this if you want to use multiple
            context sets. In that case, set this equal to a tuple of integers
            indicating the respective output dimensionalities.
        dim_yt (int, optional): Dimensionality of the outputs of the target set. You
            should set this if the dimensionality of the outputs of the target set is
            not equal to the dimensionality of the outputs of the context set.
        dim_embedding (int, optional): Dimensionality of the embedding. Defaults to 128.
        attention (bool, optional): Use attention for the deterministic encoder.
            Defaults to `False`.
        attention_num_heads (int, optional): Number of heads. Defaults to `8`.
        num_enc_layers (int, optional): Number of layers in the encoder. Defaults to 3.
        num_dec_layers (int, optional): Number of layers in the decoder. Defaults to 6.
        width (int, optional): Widths of all intermediate MLPs. Defaults to 512.
        likelihood (str, optional): Likelihood. Must be one of `"het"` or `"lowrank"`.
            Defaults to `"lowrank"`.
        num_basis_functions (int, optional): Number of basis functions for the
            low-rank likelihood. Defaults to 512.
        dim_lv (int, optional): Dimensionality of the latent variable. Defaults to 0.
        lv_likelihood (str, optional): Likelihood of the latent variable. Must be one of
            `"het"` or `"dense"`. Defaults to `"het"`.
        transform (str or tuple[float, float]): Bijection applied to the
            output of the model. This can help deal with positive of bounded data.
            Must be either `"positive"`, `"exp"`, or `"softplus"` for positive data or
            `(lower, upper)` for data in this open interval.
        dtype (dtype, optional): Data type.

    Returns:
        :class:`.model.Model`: GNP model.
    """
    # Make sure that `dim_yc` is initialised and a tuple.
    dim_yc = convert(dim_yc or dim_y, tuple)
    # Make sure that `dim_yt` is initialised.
    dim_yt = dim_yt or dim_y

    mlp_out_channels, selector, likelihood = construct_likelihood(
        nps,
        spec=likelihood,
        dim_y=dim_yt,
        num_basis_functions=num_basis_functions,
        dtype=dtype,
    )

    # Construct the deterministic encoder.
    if attention:
        det_encoder = nps.Parallel(
            *(
                nps.Chain(
                    nps.RepeatForAggregateInputs(
                        nps.Attention(
                            dim_x=dim_x,
                            dim_y=dim_yci,
                            dim_embedding=dim_embedding,
                            num_heads=attention_num_heads,
                            num_enc_layers=num_enc_layers,
                            dtype=dtype,
                        ),
                    ),
                    nps.DeterministicLikelihood(),
                )
                for dim_yci in dim_yc
            ),
        )
    else:
        det_encoder = nps.Parallel(
            *(
                nps.Chain(
                    nps.DeepSet(
                        nps.MLP(
                            in_dim=dim_x + dim_yci,
                            out_dim=dim_embedding,
                            num_layers=num_enc_layers,
                            width=width,
                            dtype=dtype,
                        )
                    ),
                    nps.DeterministicLikelihood(),
                )
                for dim_yci in dim_yc
            ),
        )

    # Possibly construct the stochastic encoder.
    if dim_lv > 0:
        lv_mlp_out_channels, _, lv_likelihood = construct_likelihood(
            nps,
            spec=lv_likelihood,
            dim_y=dim_lv,
            num_basis_functions=num_basis_functions,
            dtype=dtype,
        )
        lv_encoder = nps.Chain(
            nps.Parallel(
                *(
                    nps.DeepSet(
                        nps.MLP(
                            in_dim=dim_x + dim_yci,
                            out_dim=dim_embedding,
                            num_layers=num_enc_layers,
                            width=width,
                            dtype=dtype,
                        )
                    )
                    for dim_yci in dim_yc
                ),
            ),
            nps.Materialise(),
            nps.MLP(
                in_dim=dim_embedding * len(dim_yc),
                out_dim=lv_mlp_out_channels,
                num_layers=num_enc_layers,
                width=width * len(dim_yc),
                dtype=dtype,
            ),
            lv_likelihood,
        )

    encoder = nps.Chain(
        # We need to explicitly copy, because there will be multiple context sets in
        # parallel, which will otherwise dispatch to the wrong method.
        nps.Copy(2 + (dim_lv > 0)),
        nps.Parallel(
            nps.Chain(
                nps.RepeatForAggregateInputs(
                    nps.InputsCoder(),
                ),
                nps.DeterministicLikelihood(),
            ),
            det_encoder,
            *((lv_encoder,) if dim_lv > 0 else ()),
        ),
    )
    decoder = nps.Chain(
        nps.Materialise(),
        nps.RepeatForAggregateInputs(
            nps.Chain(
                nps.MLP(
                    in_dim=dim_x + dim_embedding * len(dim_yc) + dim_lv,
                    out_dim=mlp_out_channels,
                    num_layers=num_dec_layers,
                    width=width * len(dim_yc),
                    dtype=dtype,
                ),
                selector,  # Select the right target output.
            )
        ),
        likelihood,
        parse_transform(nps, transform=transform),
    )
    return nps.Model(encoder, decoder)
