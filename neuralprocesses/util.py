import lab as B

__all__ = ["modules", "register_module", "models", "register_model", "is_nonempty"]

modules = []  #: Registered modules.


def register_module(module):
    """Decorator to register a new module."""
    modules.append(module)
    return module


models = []  #: Registered models.


def register_model(model):
    """Decorator to register a new model."""
    models.append(model)
    return model


def is_nonempty(x):
    """Check if a tensor is not empty.

    Args:
        x (tensor): Tensor.

    Returns:
        bool: `True` if `x` is not empty, otherwise `False`.
    """
    return all([i > 0 for i in B.shape(x)])
