import lab as B

__all__ = ["AugmentedInput"]


class AugmentedInput:
    """An augmented input.

    Args:
        x (input): Input.
        augmentation (object): Augmentation.

    Attributes:
        x (input): Input.
        augmentation (object): Augmentation.
    """

    def __init__(self, x, augmentation):
        self.x = x
        self.augmentation = augmentation


@B.dispatch
def on_device(x: AugmentedInput):
    return B.on_device(x.x)


@B.dispatch
def dtype(x: AugmentedInput):
    return B.dtype(x.x)
