import lab as B

__all__ = ["modules", "register_module", "models", "register_model", "is_nonempty"]

modules = []


def register_module(module):
    modules.append(module)
    return module


models = []


def register_model(model):
    models.append(model)
    return model


def is_nonempty(x):
    return all([i > 0 for i in B.shape(x)])
