import torch
import torch.nn as nn
import torch.nn.functional as F
from .modules import FilmDecoder, ExplicitDecoder
from .fno import FNODecoder
from .conditioning import Film


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, kernel=3):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=kernel, stride=stride, padding=1, bias=False)
        self.bn1 = nn.GroupNorm(1, planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=kernel, stride=1, padding=1, bias=False)
        self.bn2 = nn.GroupNorm(1, planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion*planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion*planes, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(1, self.expansion*planes)
            )

    def forward(self, x):
        out = torch.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.gelu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.GroupNorm(1, out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.GroupNorm(1, out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.GroupNorm(1, out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet_Backbone(nn.Module):
    def __init__(self, block, num_blocks, num_channels, pool):
        super(ResNet_Backbone, self).__init__()
        self.in_channels = num_channels[0]
        self.pool = pool
        self.conv1 = nn.Conv2d(1, num_channels[0], kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.GroupNorm(1, num_channels[0])
        self.layer1 = self._make_layer(block, num_channels[0], num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, num_channels[1], num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, num_channels[2], num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, num_channels[3], num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_channels, planes, stride))
            self.in_channels = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x):
        out = F.gelu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        if self.pool:
            out = self.avgpool(out)
        return out


def get_resnet(hidden_channels=[64, 128, 256, 512], pool=True):
    return ResNet_Backbone(BasicBlock, [2, 2, 2, 2], hidden_channels, pool=pool)


class ResNet(nn.Module):
    def __init__(self, encoder, decoder, c_in=1, n_frequencies=300, conditional=False, len_conditional=None, **kwargs):
        super().__init__()

        self.encoder = get_resnet(hidden_channels=encoder.hidden_channels, pool=True)
        self.conditional = conditional
        if self.conditional is True:
            self.film = Film(len_conditional, encoder.hidden_channels[-1])
        if decoder.name == "explicit_mlp":
            self.decoder = ExplicitDecoder(in_dim=encoder.hidden_channels[-1], out_dim=n_frequencies, hidden_channels=decoder.hidden_channels)
        elif decoder.name == "fno":
            self.decoder = FNODecoder(in_dim=encoder.hidden_channels[-1], out_dim=n_frequencies, **decoder)
        elif decoder.name == "film_implicit_mlp":
            self.decoder = FilmDecoder(in_dim=encoder.hidden_channels[-1], out_dim=n_frequencies, hidden_channels_width=decoder.hidden_channels_width,
                                            hidden_channels_depth=decoder.hidden_channels_depth)
        else:
            raise NotImplementedError

    def forward(self, x, conditional=None, frequencies=None):
        x = self.encoder(x)
        if self.conditional is True:
            x = self.film(x, conditional)
        x = self.decoder(x)
        return x

class DeepONet(nn.Module):
    def __init__(self, encoder=[40, 40], decoder=[128, 128, 128, 512], image_size=81*121, n_frequencies=300, conditional=False, len_conditional=None, **kwargs):
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
                    self.film = Film(len_conditional, encoder.hidden_channels[-1])

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

    def forward(self, x, conditional=None, frequencies=None):
        if self.conditional is True:
            x = self.net(((x, conditional), self.query_frequencies.view(-1, 1).to(x.device)))
        else:
            x = self.net((x, self.query_frequencies.view(-1, 1).to(x.device)))
        return x
