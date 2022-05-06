import lab.torch  # noqa

from .nn import *
from .. import *  # noqa
from ..util import modules, models, wrapped_partial


def create_init(module):
    def __init__(self, *args, **kw_args):
        Module.__init__(self)
        module.__init__(self, *args, **kw_args)

    return __init__


def create_forward(Module):
    def forward(self, *args, **kw_args):
        return Module.__call__(self, *args, **kw_args)

    return forward


for module in modules:
    globals()[module.__name__] = type(
        module.__name__,
        (module, Module),
        {"__init__": create_init(module), "forward": create_forward(module)},
    )


class Namespace:
    pass


ns = Namespace()
ns.__dict__.update(globals())

for model in models:
    globals()[model.__name__] = wrapped_partial(model, nps=ns)
