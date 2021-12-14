import lab as B

__all__ = [
    "modules",
    "register_module",
    "models",
    "register_model",
    "batch_size",
    "feature_size",
]

modules = []


def register_module(module):
    modules.append(module)
    return module


models = []


def register_model(model):
    models.append(model)
    return model


def batch_size(x):
    return B.shape(x)[0]


def feature_size(x):
    return B.shape(x)[1]
