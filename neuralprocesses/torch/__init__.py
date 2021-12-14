import lab.torch  # noqa

from .nn import *
from .. import *  # noqa
from ..util import modules, models


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
    name = module.__name__
    globals()[name] = type(
        name,
        (module, Module),
        {"__init__": create_init(module), "forward": create_forward(module)},
    )


class Namespace:
    pass


ns = Namespace()
ns.__dict__.update(globals())

for model in models:
    name = "_".join(model.__name__.split("_")[1:])
    globals()[name] = model(ns)
