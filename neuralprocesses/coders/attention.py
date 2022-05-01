import lab as B

from .. import _dispatch
from ..util import register_module, batch

__all__ = ["Attention", "SelfAttention"]


@register_module
class Attention:
    """Attention module.

    Args:
        dim_x (int): Dimensionality of the inputs.
        dim_y (int): Dimensionality of the outputs.
        dim_embedding (int): Dimensionality of the embedding.
        num_heads (int): Number of heads.
        num_enc_layers (int): Number of layers in the encoders.
        dtype (dtype, optional): Data type.

    Attributes:
        num_heads (int): Number of heads.
        dim_head (int): Dimensionality of a head.
        encoder_x (function): Encoder for the inputs.
        encoder_xy (function): Encoder for the inputs-output pairs.
        mixer (function): Mixer.
        mlp1 (function): First MLP for the final normalisation layers.
        ln1 (function): First normaliser for the final normalisation layers.
        mlp2 (function): Second MLP for the final normalisation layers.
        ln2 (function): Second normaliser for the final normalisation layers.
    """

    def __init__(
        self,
        dim_x,
        dim_y,
        dim_embedding,
        num_heads,
        num_enc_layers,
        dtype=None,
        _self=False,
    ):
        self._self = _self

        self.num_heads = num_heads
        self.dim_head = dim_embedding // num_heads

        self.encoder_x = self.nps.MLP(
            in_dim=dim_x,
            layers=(self.dim_head * num_heads,) * num_enc_layers,
            out_dim=self.dim_head * num_heads,
            dtype=dtype,
        )
        self.encoder_xy = self.nps.MLP(
            in_dim=dim_x + dim_y,
            layers=(self.dim_head * num_heads,) * num_enc_layers,
            out_dim=self.dim_head * num_heads,
            dtype=dtype,
        )

        self.mixer = self.nps.MLP(
            in_dim=self.dim_head * num_heads,
            layers=(),
            out_dim=dim_embedding,
            dtype=dtype,
        )
        self.mlp1 = self.nps.MLP(
            in_dim=self.dim_head * num_heads,
            layers=(),
            out_dim=dim_embedding,
            dtype=dtype,
        )
        self.ln1 = self.nn.LayerNorm(dim_embedding, None, dtype=dtype)
        self.mlp2 = self.nps.MLP(
            in_dim=dim_embedding,
            layers=(dim_embedding,),
            out_dim=dim_embedding,
            dtype=dtype,
        )
        self.ln2 = self.nn.LayerNorm(dim_embedding, None, dtype=dtype)

    def _extract_heads(self, z):
        b = batch(z, 3)
        b_last, c, n = B.shape(z, -3, -2, -1)
        return B.reshape(z, *b, b_last * self.num_heads, c // self.num_heads, n)

    def _compress_heads(self, z):
        b = batch(z, 3)
        b_last, c, n = B.shape(z, -3, -2, -1)
        # Torch dimensions may not support `//`, so convert to an integer.
        return B.reshape(z, *b, b_last // self.num_heads, c * self.num_heads, n)


@_dispatch
def code(coder: Attention, xz: B.Numeric, z: B.Numeric, x: B.Numeric, **kw_args):
    if coder._self:
        # Perform self-attention rather than cross-attention.
        x = xz

    if B.shape(z, -1) == 0:
        # Handle the case of empty context set.
        queries = coder.encoder_x(x)
        with B.on_device(z):
            z = B.zeros(
                B.dtype(z),
                *batch(z, 3),
                # Return in compressed head format.
                B.shape(z, -3) * coder.num_heads,
                coder.dim_head,
                B.shape(x, -1),
            )
    else:
        keys = coder._extract_heads(coder.encoder_x(xz))
        values = coder._extract_heads(coder.encoder_xy(B.concat(xz, z, axis=-2)))
        # Don't extract heads for the queries here, because we need it is this form down
        # at the final normalisation layers.
        queries = coder.encoder_x(x)

        activations = B.matmul(keys, coder._extract_heads(queries), tr_a=True)
        activations = activations / B.sqrt(B.shape(keys, -2))  # Keep variance constant.
        z = B.matmul(values, B.softmax(activations, axis=-2))

    # Mix heads.
    z = coder.mixer(coder._compress_heads(z))

    # Apply final two residual normalisation layers.
    z = coder.ln1(z + coder.mlp1(queries))
    z = coder.ln2(z + coder.mlp2(z))

    return x, z


@register_module
class SelfAttention(Attention):
    """Self-attention module.

    Args:
        dim_x (int): Dimensionality of the inputs.
        dim_y (int): Dimensionality of the outputs.
        dim_embedding (int): Dimensionality of the embedding.
        num_heads (int): Number of heads.
        num_enc_layers (int): Number of layers in the encoders.
        dtype (dtype, optional): Data type.

    Attributes:
        num_heads (int): Number of heads.
        dim_head (int): Dimensionality of a head.
        encoder_x (function): Encoder for the inputs.
        encoder_xy (function): Encoder for the inputs-output pairs.
        mixer (function): Mixer.
        mlp1 (function): First MLP for the final normalisation layers.
        ln1 (function): First normaliser for the final normalisation layers.
        mlp2 (function): Second MLP for the final normalisation layers.
        ln2 (function): Second normaliser for the final normalisation layers.
    """

    def __init__(self, *args, **kw_args):
        Attention.__init__(self, *args, _self=True, **kw_args)
