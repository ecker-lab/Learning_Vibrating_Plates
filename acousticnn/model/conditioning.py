import torch.nn as nn


class Film(nn.Module):
    def __init__(self, conditional_dim, projection_dim, **kwargs):
        super().__init__()
        self.weight = nn.Sequential(nn.Linear(conditional_dim, projection_dim, bias=False),
                                    nn.ReLU(),
                                    nn.Linear(projection_dim, projection_dim))
        self.bias = nn.Sequential(nn.Linear(conditional_dim, projection_dim, bias=False),
                                    nn.ReLU(),
                                    nn.Linear(projection_dim, projection_dim))

    def forward(self, x, conditional):
        ndim = len(x.shape) - 2
        view_shape = (x.shape[:2]) + (1,) * ndim
        return self.weight(conditional).view(*view_shape) * x + self.bias(conditional).view(*view_shape)
