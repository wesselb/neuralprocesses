import neuralprocesses as nps  # This fixes inspection below.

from ..util import register_model

__all__ = ["construct_agnp"]


@register_model
def construct_agnp(*args, nps=nps, num_heads=8, **kw_args):
    """An Attentive Gaussian Neural Process.

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
        num_heads (int, optional): Number of heads. Defaults to `8`.
        num_enc_layers (int, optional): Number of layers in the encoder. Defaults to 3.
        enc_same (bool, optional): Use the same encoder for all context sets. This
            only works if all context sets have the same dimensionality. Defaults to
            `False`.
        num_dec_layers (int, optional): Number of layers in the decoder. Defaults to 6.
        width (int, optional): Widths of all intermediate MLPs. Defaults to 512.
        likelihood (str, optional): Likelihood. Must be one of `"het"` or `"lowrank"`.
            Defaults to `"lowrank"`.
        num_basis_functions (int, optional): Number of basis functions for the
            low-rank likelihood. Defaults to 512.
        dim_lv (int, optional): Dimensionality of the latent variable. Defaults to 0.
        lv_likelihood (str, optional): Likelihood of the latent variable. Must be one of
            `"het"`, `"dense"`, or `"spikes-beta"`. Defaults to `"het"`.
        transform (str or tuple[float, float]): Bijection applied to the
            output of the model. This can help deal with positive of bounded data.
            Must be either `"positive"`, `"exp"`, `"softplus"`, or
            `"softplus_of_square"` for positive data or `(lower, upper)` for data in
            this open interval.
        dtype (dtype, optional): Data type.

    Returns:
        :class:`.model.Model`: AGNP model.
    """
    return nps.construct_gnp(
        *args,
        nps=nps,
        attention=True,
        attention_num_heads=num_heads,
        **kw_args,
    )
