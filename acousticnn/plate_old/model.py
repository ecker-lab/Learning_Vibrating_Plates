import torch
import torch.nn as nn
from acousticnn.model import ResNet18, get_resnet, get_vit
from acousticnn.model import UNet, Film, QueryUNet, FNODecoder, FNO


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


class ImplicitDecoder(nn.Module):
    def __init__(self, in_dim=1024, out_dim=300, hidden_channels_width=[256], hidden_channels_depth=4):
        super().__init__()
        hidden_channels = hidden_channels_width * hidden_channels_depth
        self.decoder = MLP(in_dim, hidden_channels=hidden_channels + [1], norm_layer=None)
        self.query_frequencies = torch.linspace(-1, 1, out_dim).float()
        self.out_dim = out_dim

    def redefine_out_dim(self, out_dim):
        self.query_frequencies = torch.linspace(-1, 1, out_dim).float()
        self.out_dim = out_dim

    def forward(self, x):
        B = x.shape[0]
        x = x.reshape(B, -1)
        x = x.repeat_interleave(len(self.query_frequencies), dim=0)  # B*200 x num_parameters
        x = torch.hstack((x, self.query_frequencies.repeat(B).view(-1, 1).to(x.device)))
        x = self.decoder(x)
        # TODO Can I use einsum to do this more efficiently? x = torch.einsum("bi,ni->bn", x_func, x_loc)

        return x.view(B, self.out_dim)


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


class ResNet(nn.Module):
    def __init__(self, encoder, decoder, c_in=1, n_frequencies=300, conditional=False, **kwargs):
        super().__init__()

        self.encoder = get_resnet(hidden_channels=encoder.hidden_channels, pool=True)
        self.conditional = conditional
        if self.conditional is True:
            self.film = Film(4, encoder.hidden_channels[-1])
        if decoder.name == "implicit_mlp":
            self.decoder = ImplicitDecoder(in_dim=encoder.hidden_channels[-1] + 1, out_dim=n_frequencies, hidden_channels_width=decoder.hidden_channels_width, 
                                            hidden_channels_depth=decoder.hidden_channels_depth)
        elif decoder.name == "explicit_mlp":
            self.decoder = ExplicitDecoder(in_dim=encoder.hidden_channels[-1], out_dim=n_frequencies, hidden_channels=decoder.hidden_channels)
        elif decoder.name == "fno":
            self.decoder = FNODecoder(in_dim=encoder.hidden_channels[-1], out_dim=n_frequencies, **decoder)
        else:
            raise NotImplementedError

    def forward(self, x, conditional=None):
        x = self.encoder(x)
        if self.conditional is True:
            x = self.film(x, conditional)
        x = self.decoder(x)
        return x


class VIT(nn.Module):
    def __init__(self, encoder, decoder, c_in=1, n_frequencies=300, conditional=False, **kwargs):
        super().__init__()

        self.encoder = get_vit(encoder.hidden_dim_size, pool=True)
        self.conditional = conditional

        if self.conditional is True:
            self.film = Film(4, encoder.hidden_dim_size)
        if decoder.name == "implicit_mlp":
            self.decoder = ImplicitDecoder(in_dim=encoder.hidden_dim_size + 1, out_dim=n_frequencies, hidden_channels_width=decoder.hidden_channels_width, 
                                            hidden_channels_depth=decoder.hidden_channels_depth)
        else:
            raise NotImplementedError

    def forward(self, x, conditional=None):
        x = self.encoder(x)
        if self.conditional is True:
            x = self.film(x, conditional)
        x = self.decoder(x)
        return x


class DeepONet(nn.Module):
    def __init__(self, encoder=[40, 40], decoder=[128, 128, 128, 512], image_size=81*121, n_frequencies=300, conditional=False, **kwargs):
        super().__init__()
        import deepxde as dde
        torch.set_default_tensor_type('torch.FloatTensor')
        self.conditional = conditional

        class DeepOnetEncoder(nn.Module):
            def __init__(self, encoder, conditional=False):
                super().__init__()
                self.conditional = conditional
                self.encoder = get_resnet(hidden_channels=encoder.hidden_channels, pool=True)     
                if self.conditional is True:
                    self.film = Film(4, encoder.hidden_channels[-1])

            def forward(self, x):
                if self.conditional is True:
                    x, conditional = x[0], x[1]
                x = self.encoder(x)
                if self.conditional is True:
                    x = self.film(x, conditional)
                return x.flatten(start_dim=1)
                    
        self.encoder = DeepOnetEncoder(encoder, conditional)
        self.net = dde.nn.pytorch.deeponet.DeepONetCartesianProd(
            layer_sizes_branch=[128, self.encoder],
            layer_sizes_trunk=[1] + decoder,
            activation="relu",
            kernel_initializer="Glorot normal")
        self.query_frequencies = torch.linspace(-1, 1, n_frequencies).float()
        self.out_dim = n_frequencies

    def forward(self, x, conditional=None):
        if self.conditional is True:
            x = self.net(((x, conditional), self.query_frequencies.view(-1, 1).to(x.device)))
        else:
            x = self.net((x, self.query_frequencies.view(-1, 1).to(x.device)))
        return x


def model_factory(model_name, **kwargs):
    if model_name == 'ResNet':
        return ResNet(**kwargs)
    if model_name == 'vit':
        return VIT(**kwargs)
    # Add additional elif statements here as you add more models
    if model_name == "DeepONet":
        return DeepONet(**kwargs)
    if model_name == "UNet":
        return UNet(**kwargs)
    if model_name == "QueryUNet":
        return QueryUNet(**kwargs)
    if model_name == "FNO":
        return FNO(**kwargs)

    else:
        raise ValueError(f'Unknown model: {model_name}')
