import lab as B
from matrix.util import indent
from plum import List, Tuple, Union

from . import _dispatch
from .augment import AugmentedInput
from .coding import code
from .dist import AbstractMultiOutputDistribution
from .mask import Masked
from .parallel import Parallel
from .util import register_module

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
    def __call__(self, xc, yc, xt, *, num_samples=1, aux_t=None, **kw_args):
        """Run the model.

        Args:
            xc (input): Context inputs.
            yc (tensor): Context outputs.
            xt (input): Target inputs.
            num_samples (int, optional): Number of samples, if applicable. Defaults
                to 1.
            aux_t (tensor, optional): Target-specific auxiliary input, if applicable.

        Returns:
            tuple[input, tensor]: Target inputs and prediction for target outputs.
        """
        # TODO: Handle random state.

        # Perform augmentation of `xt` with auxiliary target information.
        if aux_t is not None:
            xt = AugmentedInput(xt, aux_t)

        xz, z = code(self.encoder, xc, yc, xt, **kw_args)
        z = _sample(z, num=num_samples)
        _, d = code(self.decoder, xz, z, xt, **kw_args)

        return d

    @_dispatch
    def __call__(
        self,
        contexts: List[Tuple[Union[B.Numeric, tuple], Union[B.Numeric, Masked]]],
        xt,
        **kw_args,
    ):
        return self(
            Parallel(*(c[0] for c in contexts)),
            Parallel(*(c[1] for c in contexts)),
            xt,
            **kw_args,
        )

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


@_dispatch
def _sample(x: AbstractMultiOutputDistribution, num: B.Int = 1):
    return x.sample(num=num)


@_dispatch
def _sample(x: Parallel, num: B.Int = 1):
    return Parallel(*[_sample(xi, num=num) for xi in x])
