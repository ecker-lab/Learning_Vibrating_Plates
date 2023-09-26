import math
import torch
from torch import nn
import numpy as np
import math

def pos_enc_sincos(x, dim=10, base=10000, factor=1):
    """ 
    sinosoidal positional encoding 
    important: value range of x must be considered. This configuration works best in [-100, 100].
    """

    old_shape = x.shape
    x = x.flatten()
    div_vec = torch.exp(torch.arange(0, dim, 2).to(x.device) * (-math.log(base) / dim))

    pe2 = torch.cat([
        torch.sin(x[:,None] * div_vec[None, :]*factor),
        torch.cos(x[:,None] * div_vec[None, :]*factor),
    ], dim=1)

    pe2 = pe2.view(old_shape + (dim,))

    return pe2


class SinosoidalEncoding(nn.Module):
    def __init__(self, dim=128, base=10000, factor=1):
        """
        :param dim: the encoding dimension 
        :base_param: parameter to scale the frequency of the sinosoidal functions 
        """
        super().__init__()
        self.dim = dim
        self.base =  base
        self.factor = factor

    def forward(self, tensor):
        # Input: tensor of shape B x num tokens
        # Output: tensor of shape B x num_tokens x dim
        return pos_enc_sincos(tensor, self.dim, self.base, self.factor)


class PositionalEncoding(nn.Module):
    def __init__(self, dim=128, base=0.3):
        """
        :param dim: the encoding dimension 
        :base_param: parameter to scale the frequency of the sinosoidal functions 
        """
        super().__init__()
        self.encoder = SinosoidalEncoding(dim, base)
        self.cached_penc = None

    def forward(self, tensor):
        """
        Input: tensor of shape B x num tokens
        Output: tensor of shape B x num_tokens x dim
        """
        if self.cached_penc is not None and self.cached_penc.shape == tensor.shape:
            return self.cached_penc

        self.cached_penc = None
        batch_size, num_tokens = tensor.shape
        pos_x = torch.arange(num_tokens, device=tensor.device).float().unsqueeze(0).repeat(batch_size, 1)
        self.cached_penc = self.encoder(pos_x)
        
        return self.cached_penc


class FourierEncoding(nn.Module):
    def __init__(self, dim=16, n_parameters=2, factor_pars=(0, 1), factor=2*torch.pi):
        """ 
        # choose factor as a random value from a normal distribution matrix multiplication
        """
        super().__init__()
        self.dim=dim
        shape = (n_parameters, int(dim/2)*n_parameters)
        self.matrix = torch.from_numpy(np.random.normal(*factor_pars, shape)).float()
        self.factor = factor

    def forward(self, x):
        old_shape = x.shape
        self.factor = self.matrix.to(x.device)
        encoding = torch.cat([
            torch.sin(self.factor * x[:,None] @ self.matrix),
            torch.cos(self.factor * x[:,None] @ self.matrix),
        ], dim=1)
        encoding = encoding.view((*old_shape, self.dim))

        return encoding


class RandomEncoding(nn.Module):
    def __init__(self, dim=16, factor_pars = (0, 1)):
        """ 
        # choose factor as a random value from a normal distribution only one factor
        """
        super().__init__()
        self.dim=dim
        shape = (int(dim/2))
        div_vec = torch.from_numpy(np.random.normal(*factor_pars, shape)).float()
        self.register_buffer("div_vec", div_vec)
    def forward(self, x):
        old_shape = x.shape
        x = x.flatten()
        self.div_vec = self.div_vec.to(x.device)
        encoding = torch.cat([
            torch.sin(x[:,None] * self.div_vec[None, :]),
            torch.cos(x[:,None] * self.div_vec[None, :]),
        ], dim=1)

        encoding = encoding.view((*old_shape, self.dim))

        return encoding
        
