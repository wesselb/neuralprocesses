__all__ = ["data", "register_data"]


data = {}  #: All data sets to train on


def register_data(name, setup, requires_dim_x=False, requires_dim_y=False):
    """Register a data set.

    Args:
        name (str): Name.
        setup (function): Setup function.
        requires_dim_x (bool, optional): Requires the value of `--dim-x`. Defaults to
            `False`.
        requires_dim_y (bool, optional): Requires the value of `--dim-y`. Defaults to
            `False`.
    """
    data[name] = {
        "setup": setup,
        "requires_dim_x": requires_dim_x,
        "requires_dim_y": requires_dim_y,
    }
