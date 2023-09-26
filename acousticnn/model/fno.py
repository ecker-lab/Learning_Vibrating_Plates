import torch.nn as nn
import torch
import torch.nn.functional as F
from acousticnn.model import Film
from neuralop.models import TFNO
from neuralop.models import FNO as FNOBlock


class FNOConditionalBlock(FNOBlock):
    def __init__(self, hidden_channels, conditional, **kwargs):
        super().__init__(hidden_channels=hidden_channels, **kwargs)
        self.film = Film(4, hidden_channels)
        self.conditional = conditional

    def forward(self, x, conditional=None):
        """TFNO's forward pass
        """
        x = self.lifting(x)

        if self.domain_padding is not None:
            x = self.domain_padding.pad(x)

        for layer_idx in range(self.n_layers):
            x = self.fno_blocks(x, layer_idx)
            if self.conditional is True:
                if layer_idx == 1:
                    x = self.film(x, conditional)

        if self.domain_padding is not None:
            x = self.domain_padding.unpad(x)

        x = self.projection(x)

        return x


class FNO(nn.Module):
    def __init__(self, c_in=3, c_out=300, conditional=False, n_modes=16, hidden_channels=64, n_layers=2, tfno=False, film=True, **kwargs):
        super().__init__()
        self.conditional = conditional
        self.film = film
        if self.film is True and self.conditional is True:
            self.fno1 = FNOConditionalBlock(n_modes=[n_modes, n_modes], in_channels=1 + 2, hidden_channels=hidden_channels,
                                            out_channels=c_out, n_layers=n_layers*2, conditional=conditional)
        elif self.conditional is True:
            in_channels = 3 + self.conditional * 4
            self.fno1 = FNOBlock(n_modes=[n_modes, n_modes], in_channels=in_channels, hidden_channels=hidden_channels, out_channels=c_out, n_layers=4)
        else:
            in_channels = 3 
            self.fno1 = FNOBlock(n_modes=[n_modes, n_modes], in_channels=in_channels, hidden_channels=hidden_channels, out_channels=c_out, n_layers=4)

    @staticmethod
    def add_positional_encoding(tensor):
        B, c, w, h = tensor.shape
        positional_encoding_x = torch.linspace(0, 1, w, dtype=tensor.dtype, device=tensor.device).view(1, 1, w, 1).repeat(B, 1, 1, h)
        positional_encoding_y = torch.linspace(0, 1, h, dtype=tensor.dtype, device=tensor.device).view(1, 1, 1, h).repeat(B, 1, w, 1)
        tensor_with_pos_encoding = torch.cat((tensor, positional_encoding_x, positional_encoding_y), dim=1)
        return tensor_with_pos_encoding

    def forward(self, x, conditional=None):
        x = F.interpolate(x, (40, 60))

        if self.conditional is True and self.film is False:
            conditional_reshaped = conditional.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, x.shape[2], x.shape[3])
            x = torch.cat([x, conditional_reshaped], dim=1)  # Resulting shape will be (B, 5, W, H)
        x = self.add_positional_encoding(x)
        if self.film is True and self.conditional is True:
            x = self.fno1(x, conditional)
        else:
            x = self.fno1(x)
        return x



class FNODecoder(nn.Module):
    def __init__(self, hidden_channels, n_modes, n_layers, tfno, in_dim=1024, out_dim=300, conditional=False, **kwargs):
        super().__init__()
        self.out_dim = out_dim
        self.conditional = conditional
        self.linear = torch.nn.Linear(in_dim, out_dim)
        if tfno is True:
            self.decoder = TFNO(n_modes=[n_modes], in_channels=2, hidden_channels=hidden_channels, n_layers=n_layers, factorization='tucker', rank=0.05, implementation="factorized")
        elif self.conditional is False:
            self.decoder = FNOBlock(n_modes=[n_modes], in_channels=2, hidden_channels=hidden_channels, n_layers=n_layers)
        else:
            self.decoder = FNOConditionalBlock(n_modes=[n_modes, n_modes], in_channels=1 + 2, hidden_channels=hidden_channels,
                                            out_channels=c_out, n_layers=n_layers*2, conditional=conditional)
    @staticmethod
    def add_positional_encoding(tensor):
        B, d = tensor.shape
        positional_encoding = torch.linspace(0, 1, d, dtype=tensor.dtype, device=tensor.device).view(1, d, 1) / d
        positional_encoding = positional_encoding.repeat(B, 1, 1)
        tensor_with_pos_encoding = torch.cat((tensor.unsqueeze(-1), positional_encoding), dim=-1)
        return tensor_with_pos_encoding

    def forward(self, x, conditional=None):
        B = x.shape[0]
        x = x.reshape(B, -1)
        x = self.linear(x)
        x = self.add_positional_encoding(x)
        if self.conditional is True:
            x = sself.decoder(x.transpose(1, 2), conditional)
        else:
            x = self.decoder(x.transpose(1, 2))

        return x.view(B, self.out_dim)

