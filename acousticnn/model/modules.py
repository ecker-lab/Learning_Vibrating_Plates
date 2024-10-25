import torch
import torch.nn as nn
from .conditioning import Film


class MLP(nn.Module):
    def __init__(self, input_size, hidden_channels, act_layer=nn.ReLU, norm_layer=None):
        super(MLP, self).__init__()

        layers = []
        last_size = input_size
        for hidden_size in hidden_channels[:-1]:
            layers.append(nn.Linear(last_size, hidden_size))
            if norm_layer is not None:
                layers.append(norm_layer(1, hidden_size))
            layers.append(act_layer())
            last_size = hidden_size

        layers.append(nn.Linear(last_size, hidden_channels[-1]))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class FilmDecoder(nn.Module):
    def __init__(self, in_dim=1024, out_dim=300, hidden_channels_width=[256], hidden_channels_depth=4):
        super().__init__()
        hidden_channels = hidden_channels_width * hidden_channels_depth
        self.decoder = MLP(in_dim, hidden_channels=hidden_channels + [1], norm_layer=None)
        self.query_frequencies = torch.linspace(-1, 1, out_dim).float()
        self.out_dim = out_dim
        self.queryfilm = Film(1, in_dim)

    def redefine_out_dim(self, out_dim):
        self.query_frequencies = torch.linspace(-1, 1, out_dim).float()
        self.out_dim = out_dim

    def forward(self, x, query_filter_fn=None):
        B = x.shape[0]
        x = x.reshape(B, -1)
        if query_filter_fn is not None:
            queries = query_filter_fn(self.query_frequencies)
        else:
            queries = self.query_frequencies
        x = x.repeat_interleave(len(queries), dim=0)
        qf = queries.repeat(B).view(-1, 1).to(x.device)
        x = self.queryfilm(x, qf)
        x = self.decoder(x)
        return x.view(B, len(queries))


class ExplicitDecoder(nn.Module):
    def __init__(self, in_dim=1024, out_dim=300, hidden_channels=4*[512]):
        super().__init__()
        self.out_dim = out_dim
        self.decoder = MLP(in_dim, hidden_channels=hidden_channels + [self.out_dim])

    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B, -1)
        x = self.decoder(x)

        return x.view(B, self.out_dim)