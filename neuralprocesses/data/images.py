import lab as B
from .data import DataGenerator
from ..dist import UniformDiscrete

import torch
import torchvision
import lab as B

from torchvision.transforms import ToTensor



__all__ = ["ImageGenerator"]


class ImageGenerator(DataGenerator):
    """
    """

    allowed_datasets = {
        "mnist"  : "MNIST",
        "svhn"   : "SVHN",
        "celeba" : "CelebA",
    }

    @B.dispatch
    def __init__(
            self,
            dataset: str,
            data_root: str,
            split: str,
            batch_size: int,
            dtype: torch.dtype,
            device: str,
            seed: int,
            num_tasks: int,
            num_targets: UniformDiscrete = UniformDiscrete(100, 256),
            *args,
            **kw_args
        ):

        # Check dataset is one of the allowed datasets
        if dataset not in self.datasets:

            raise ValueError(
                f"Dataset {dataset} is not in the list of allowed datasets:"
                f" {self.allowed_datasets.keys()}"
            )

        # Check the split is one of the allowed splits
        if split not in ["train", "test"]:

            raise ValueError(
                f"Dataloader supports splits 'train' and 'test'."
                f" {split} is not supported."
            )
            
        # Do superclass initialisation
        super().__init__(
            dtype=dtype,
            seed=seed,
            num_tasks=num_tasks,
            batch_size=batch_size,
            device=device,
        )

        # Set the number of targets
        self.num_targets = num_targets

        # Set torch dataset
        torch_dataset = getattr(
            torchvision.datasets,
            self.allowed_datasets[dataset]
        )

        self.dataset = torch_dataset(
            root=data_root,
            transform=ToTensor()
        )

        # Initialise dataloader
        self.dataloader = torch.utils.data.DataLoader(
            dataset=self.dataset,
            batch_size=batch_size,
            shuffle=True,
            train=(split == "train")
        )


    def epoch(self):
        self.iterator = self.dataloader.__iter__()
        return super().epoch()

    def generate_batch(self):

        images, _ = self.iterator.__next__()

        with B.on_device(self.device):
            contexts = [
                (
                    B.to_active_device(B.cast(self.dtype, x)),
                    B.to_active_device(B.cast(self.dtype, y)),
                )
                for x, y in contexts
            ]

            xt = AggregateTargets(
                *[
                    (B.to_active_device(B.cast(self.dtype, _xt)), i)
                    for _xt, i in xt
                ]
            )

            yt = Aggregate(*[B.to_active_device(B.cast(self.dtype, _yt)) for _yt in yt])

        batch = {
            "contexts": contexts,
            "xt": xt,
            "yt": yt,
        }

        return batch
