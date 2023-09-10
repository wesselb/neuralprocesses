import functools

import plum

_internal_dispatch = plum.Dispatcher()


def _dispatch(f=None, **kw_args):
    if f is None:
        return functools.partial(_dispatch, **kw_args)

    if f.__name__ in {"code", "code_track", "recode"}:

        @functools.wraps(f)
        def f_wrapped(*args, **kw_args):
            if "root" not in kw_args or not kw_args["root"]:
                raise RuntimeError(
                    "Did you not set `root = True` at the root coding call, "
                    "or did you forget to propagate `**kw_args`?"
                )
            return f(*args, **kw_args)

        return _internal_dispatch(f_wrapped, **kw_args)
    else:
        return _internal_dispatch(f, **kw_args)


from .aggregate import *
from .architectures import *
from .augment import *
from .chain import *
from .coders import *
from .coding import *
from .data import *
from .datadims import *
from .disc import *
from .dist import *
from .likelihood import *
from .mask import *
from .materialise import *
from .model import *
from .numdata import *
from .parallel import *
