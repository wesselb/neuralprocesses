import lab.tensorflow  # noqa
from plum import convert

from .nn import *
from .. import *  # noqa
from ..util import modules, models, wrapped_partial


def create_init(module):
    def __init__(self, *args, **kw_args):
        Module.__init__(self)
        module.__init__(self, *args, **kw_args)

    return __init__


def create_tf_call(module):
    def call(self, *args, training=False, **kw_args):
        try:
            return module.__call__(self, *args, training=training, **kw_args)
        except TypeError:
            return module.__call__(self, *args, **kw_args)

    return call


for module in modules:
    globals()[module.__name__] = type(
        module.__name__,
        (module, Module),
        {
            "__init__": create_init(module),
            "__call__": create_tf_call(module),
            "call": create_tf_call(module),
        },
    )


class Namespace:
    pass


ns = Namespace()
ns.__dict__.update(globals())

for model in models:
    globals()[model.__name__] = wrapped_partial(model, nps=ns)
