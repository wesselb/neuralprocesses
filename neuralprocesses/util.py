import lab as B

__all__ = [
    "abstract_modules",
    "abstract",
    "model_constructors",
    "register_model",
    "batch_size",
    "feature_size",
]

abstract_modules = []


def abstract(module):
    abstract_modules.append(module)
    return module


model_constructors = []


def register_model(model):
    model_constructors.append(model)
    return model


def batch_size(x):
    return B.shape(x)[0]


def feature_size(x):
    return B.shape(x)[1]
