from torch import nn
from transformers import ViTConfig, ViTModel
from torchvision import transforms
from .conditioning import Film
from .modules import FilmDecoder

class CustomViT(nn.Module):
    def __init__(self, config, pool):
        super(CustomViT, self).__init__()
        self.pool = pool
        self.resizer = transforms.Resize((96, 128))

        self.vit = ViTModel(config)
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.linear = nn.Linear(96*128, 192)
    def forward(self, x_0):
        x_0 = self.resizer(x_0)
        x = self.vit(x_0).last_hidden_state
        if self.pool:
            x = self.avgpool(x.permute(0, 2, 1)) + self.linear(x_0.view(-1, 96*128)).unsqueeze(-1)
        return x


def get_vit(hidden_dim_size, pool=True):
    config = ViTConfig()
    config.image_size = (96, 128)
    config.layer_norm_eps = 1e-12
    config.model_type = "vit"
    config.num_attention_heads = 3
    config.num_hidden_layers = 12
    config.patch_size = 16
    config.qkv_bias = True
    config.hidden_act = "gelu"
    config.hidden_dropout_prob = 0.0
    config.intermediate_size = 768
    config.hidden_size = hidden_dim_size
    config.num_channels = 1
    model = CustomViT(config, pool)
    return model


class VIT(nn.Module):
    def __init__(self, encoder, decoder, c_in=1, n_frequencies=300, conditional=False, len_conditional=None, **kwargs):
        super().__init__()

        self.encoder = get_vit(encoder.hidden_dim_size, pool=True)
        self.conditional = conditional

        if self.conditional is True:
            self.film = Film(len_conditional, encoder.hidden_dim_size)
        self.decoder = FilmDecoder(in_dim=encoder.hidden_dim_size, out_dim=n_frequencies, hidden_channels_width=decoder.hidden_channels_width,
                                        hidden_channels_depth=decoder.hidden_channels_depth)

    def forward(self, x, conditional=None, frequencies=None):
        x = self.encoder(x)
        if self.conditional is True:
            x = self.film(x, conditional)
        x = self.decoder(x)
        return x