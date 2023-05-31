from typing import List, Tuple, Union

import lab as B
from matrix.util import indent

from .. import _dispatch
from ..augment import AugmentedInput
from ..coding import code
from ..mask import Masked
from ..util import register_module
from .util import compress_contexts, sample

__all__ = ["Model"]


@register_module
class Model:
    """Encoder-decoder model.

    Args:
        encoder (coder): Coder.
        decoder (coder): Coder.

    Attributes:
        encoder (coder): Coder.
        decoder (coder): Coder.
    """

    def __init__(self, encoder, decoder):
        self.encoder = encoder
        self.decoder = decoder

    @_dispatch
    def __call__(
        self,
        state: B.RandomState,
        xc,
        yc,
        xt,
        *,
        num_samples=None,
        aux_t=None,
        dtype_enc_sample=None,
        **kw_args,
    ):
        """Run the model.

        Args:
            state (random state, optional): Random state.
            xc (input): Context inputs.
            yc (tensor): Context outputs.
            xt (input): Target inputs.
            num_samples (int, optional): Number of samples, if applicable.
            aux_t (tensor, optional): Target-specific auxiliary input, if applicable.
            dtype_enc_sample (dtype, optional): Data type to convert the sampled
                encoding to.

        Returns:
            random state, optional: Random state.
            input: Target inputs.
            object: Prediction for target outputs.
        """
        # Perform augmentation of `xt` with auxiliary target information.
        if aux_t is not None:
            xt = AugmentedInput(xt, aux_t)

        # If the keyword `noiseless` is set to `True`, then that only applies to the
        # decoder.
        enc_kw_args = dict(kw_args)
        if "noiseless" in enc_kw_args:
            del enc_kw_args["noiseless"]
        xz, pz = code(self.encoder, xc, yc, xt, root=True, **enc_kw_args)

        # Sample and convert sample to the right data type.
        shape = () if num_samples is None else (num_samples,)
        state, z = sample(state, pz, *shape)
        if dtype_enc_sample:
            z = B.cast(dtype_enc_sample, z)

        _, d = code(self.decoder, xz, z, xt, root=True, **kw_args)

        return state, d

    @_dispatch
    def __call__(self, xc, yc, xt, **kw_args):
        state = B.global_random_state(B.dtype(xt))
        state, d = self(state, xc, yc, xt, **kw_args)
        B.set_global_random_state(state)
        return d

    @_dispatch
    def __call__(
        self,
        state: B.RandomState,
        contexts: List[
            Tuple[Union[None, B.Numeric, tuple], Union[None, B.Numeric, Masked]],
        ],
        xt,
        **kw_args,
    ):
        return self(
            state,
            *compress_contexts(contexts),
            xt,
            **kw_args,
        )

    @_dispatch
    def __call__(
        self,
        contexts: List[
            Tuple[Union[None, B.Numeric, tuple], Union[None, B.Numeric, Masked]]
        ],
        xt,
        **kw_args,
    ):
        state = B.global_random_state(B.dtype(xt))
        state, d = self(state, contexts, xt, **kw_args)
        B.set_global_random_state(state)
        return d

    def __str__(self):
        return (
            f"Model(\n"
            + indent(str(self.encoder), " " * 4)
            + ",\n"
            + indent(str(self.decoder), " " * 4)
            + "\n)"
        )

    def __repr__(self):
        return (
            f"Model(\n"
            + indent(repr(self.encoder), " " * 4)
            + ",\n"
            + indent(repr(self.decoder), " " * 4)
            + "\n)"
        )
