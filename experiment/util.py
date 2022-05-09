import lab as B

__all__ = ["with_err"]


def with_err(vals, and_lower=False, and_upper=False):
    """Print the mean value of a list of values with error."""
    vals = B.to_numpy(vals)
    mean = B.mean(vals)
    err = 1.96 * B.std(vals) / B.sqrt(B.length(vals))
    res = f"{mean:10.5f} +- {err:10.5f}"
    if and_lower:
        res += f" ({mean - err:10.5f})"
    if and_upper:
        res += f" ({mean + err:10.5f})"
    return res
