from functools import wraps

__all__ = ["cache"]


def cache(f):
    """A decorator that caches the output of a function. It assumes that all arguments
    and keyword argument"""
    _f_cache = {}

    @wraps(f)
    def f_wrapped(*args, **kw_args):
        cache_key = (args, frozenset(kw_args.items()))
        try:
            return _f_cache[cache_key]
        except KeyError:
            # Cache miss. Perform computation.
            _f_cache[cache_key] = f(*args, **kw_args)
            return _f_cache[cache_key]

    return f_wrapped
