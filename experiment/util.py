import lab as B

__all__ = ["with_err"]


def with_err(vals):
    """Print the mean value of a list of values with error."""
    vals = B.to_numpy(vals)
    mean = B.mean(vals)
    err = 1.96 * B.std(vals) / B.sqrt(B.length(vals))
    return f"{mean:10.5f} +- {err:10.5f} ({mean - err:10.5f})"
