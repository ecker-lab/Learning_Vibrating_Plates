import torch

from torch import nn
from transformers import ViTConfig, ViTModel
from torchvision import transforms


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
