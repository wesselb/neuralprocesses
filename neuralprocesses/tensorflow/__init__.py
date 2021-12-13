import lab.tensorflow  # noqa

from .nn import Module
from .. import *  # noqa
from ..util import abstract_modules, model_constructors


def create_init(module):
    def __init__(self, *args, **kw_args):
        Module.__init__(self)
        module.__init__(self, *args, **kw_args)

    return __init__


def create_call(module):
    def call(self, x, training=False):
        try:
            return module.__call__(self, x, training=training)
        except TypeError:
            return module.__call__(self, x)

    return call


for module in abstract_modules:
    name = module.__name__[len("Abstract") :]
    globals()[name] = type(
        name,
        (module, Module),
        {
            "__init__": create_init(module),
            "call": create_call(module),
        },
    )


class Namespace:
    pass


ns = Namespace()
ns.__dict__.update(globals())

for model in model_constructors:
    name = "_".join(model.__name__.split("_")[1:])
    globals()[name] = model(ns)
