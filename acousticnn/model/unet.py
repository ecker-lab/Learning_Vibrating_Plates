import torch.nn as nn
import torch
import torch.nn.functional as F
from acousticnn.model import Film


# model adapted from https://github.com/dome272/Diffusion-Models-pytorch
class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        x = self.maxpool_conv(x)
        return x


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

    def forward(self, x, skip_x):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        return x


class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=300, conditional=False, scaling_factor=64, **kwargs):
        super().__init__()
        k = scaling_factor  # 32
        self.inc = DoubleConv(c_in, k)
        self.down1 = Down(k, 2*k)
        self.down2 = Down(2*k, 4*k)
        self.down3 = Down(4*k, 4*k)
        self.conditional = conditional

        if self.conditional is True:
            self.film = Film(4, 4*k)
        self.bot1 = DoubleConv(4*k, 4*k)
        self.bot3 = DoubleConv(4*k, 4*k)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.up1 = Up(12*k, 4*k)
        self.up2 = Up(6*k, 2*k)
        self.outc = nn.Conv2d(2*k, c_out, kernel_size=1)

    def forward(self, x, conditional=None):

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        if self.conditional is True:
            x3 = self.film(x3, conditional)
        x4 = self.down3(x3)

        x4 = self.bot1(x4)
        x4 = self.bot3(x4)
        gap = self.global_avg_pool(x4)
        x4 = torch.cat([x4, gap.repeat(1, 1, x4.size(2), x4.size(3))], dim=1)
        x = self.up1(x4, x3)
        x = self.up2(x, x2)
        output = self.outc(x)
        return output


class QueryUNet(nn.Module):
    def __init__(self, c_in=1, conditional=False, out_dim=300, freq_batches=5, scaling_factor=32, **kwargs):
        super().__init__()
        k = scaling_factor  # 32
        self.inc = DoubleConv(c_in, k)
        self.down1 = Down(k, 2*k)
        self.down2 = Down(2*k, 4*k)
        self.down3 = Down(4*k, 4*k)
        self.conditional = conditional

        if self.conditional is True:
            self.film = Film(4, 4*k)
        c_out = freq_batches
        self.bot1 = DoubleConv(4*k, 4*k)
        self.bot3 = DoubleConv(4*k, 4*k)
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        self.up1 = Up(12*k + 1, 4*k)
        self.up2 = Up(6*k, 2*k)
        self.outc = nn.Conv2d(2*k, c_out, kernel_size=1)
        self.freq_steps = int(out_dim / freq_batches)
        self.out_dim = out_dim
        self.query_frequencies = torch.linspace(-1, 1, self.freq_steps).cuda()

    def forward(self, x, conditional=None):
        B = x.shape[0]

        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        if self.conditional is True:
            x3 = self.film(x3, conditional)
        x4 = self.down3(x3)
        x4 = self.bot1(x4)
        x4 = self.bot3(x4)
        gap = self.global_avg_pool(x4)
        x4 = torch.cat([x4, gap.repeat(1, 1, x4.size(2), x4.size(3))], dim=1)

        x4 = x4.repeat_interleave(len(self.query_frequencies), dim=0)
        qf = self.query_frequencies.repeat(B).view(-1, 1, 1, 1).expand(B*self.freq_steps, 1, *x4.shape[2:])
        x4 = torch.cat((x4, qf), dim=1)
        x = self.up1(x4, x3.repeat_interleave(len(self.query_frequencies), dim=0))
        x = self.up2(x, x2.repeat_interleave(len(self.query_frequencies), dim=0))
        output = self.outc(x)
        return output.reshape(B, self.out_dim, *output.shape[2:])
