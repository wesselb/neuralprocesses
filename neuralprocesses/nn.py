import torch.nn
import lab as B

__all__ = ["MLP"]


class MLP(torch.nn.Module):
    def __init__(
        self,
        dim_in,
        dim_hidden,
        dim_out,
        num_layers,
        nonlinearity=torch.nn.ReLU,
    ):
        super().__init__()
        if num_layers == 1:
            layers = torch.nn.Linear(dim_in, dim_out)
        else:
            layers = [torch.nn.Linear(dim_in, dim_hidden)]
            for _ in range(num_layers - 2):
                layers.append(nonlinearity())
                layers.append(torch.nn.Linear(dim_hidden, dim_hidden))
            layers.append(nonlinearity())
            layers.append(torch.nn.Linear(dim_hidden, dim_out))
        self.layers = torch.nn.Sequential(*layers)

    def forward(self, x):
        x = B.transpose(x)
        x = self.layers(x)
        x = B.transpose(x)
        return x
