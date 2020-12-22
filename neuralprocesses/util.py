import lab as B

__all__ = ["batch_size", "feature_size"]


def batch_size(x):
    return B.shape(x)[0]


def feature_size(x):
    return B.shape(x)[1]
