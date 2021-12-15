import lab as B
from algebra.util import identical
from matrix import Dense
from plum import dispatch

from mlkernels import Kernel, pairwise, elwise


class Gibbs(Kernel):
    """Exponentiated quadratic kernel with input-dependent length scale.

    Args:
        scale (function): Length scale function.
    """

    def __init__(self, scale):
        self.scale = scale

    def _compute(self, dists2):
        return B.exp(-0.5 * dists2 / self.scale ** 2)

    def render(self, formatter):
        return f"NonstationaryEQ({formatter(self.scale)})"

    @property
    def _stationary(self):
        return False

    @dispatch
    def __eq__(self, other: "NonstationaryEQ"):
        return identical(self.scale, other.scale)


# It remains to implement pairwise and element-wise computation of the kernel.


@pairwise.dispatch
def pairwise(k: NonstationaryEQ, x: B.Numeric, y: B.Numeric):

    return Dense(k._compute(B.pw_dists2(x, y)))


@elwise.dispatch
def elwise(k: NonstationaryEQ, x: B.Numeric, y: B.Numeric):
    return k._compute(B.ew_dists2(x, y))
