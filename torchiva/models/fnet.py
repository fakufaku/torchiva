import torch
from torch import nn
import torch.nn.functional as F

from .base import SourceModelBase
from torchaudio.transforms import MelScale


class ComplexDropout(nn.Module):
    def __init__(self, p):
        super().__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x):
        if x.dtype in [torch.complex64, torch.complex128]:
            x = torch.view_as_real(x)
            x = self.dropout(x)
            return torch.view_as_complex(x)
        else:
            return self.dropout(x)


class ComplexLinear(nn.Module):
    def __init__(self, num_input, num_output, **kwargs):
        super().__init__()
        self.real = nn.Linear(num_input, num_output, **kwargs)
        self.imag = nn.Linear(num_input, num_output, **kwargs)

    def forward(self, x):

        if x.dtype in [torch.complex64, torch.complex128]:
            real = self.real(x.real) - self.imag(x.imag)
            imag = self.real(x.imag) + self.imag(x.real)
        else:
            real = self.real(x)
            imag = self.imag(x)

        out = torch.view_as_complex(
            torch.cat([real[..., None], imag[..., None]], dim=-1)
        )
        return out


def complex_non_linearity(x):
    mod = torch.clamp(x.abs(), min=1e-5)
    G = torch.sqrt(mod) / mod
    return G * x


class FeedForward(nn.Module):
    def __init__(self, num_features, expansion_factor, dropout, complex=False):
        super().__init__()
        num_hidden = expansion_factor * num_features
        self.complex = complex
        if complex:
            self.fc1 = ComplexLinear(num_features, num_hidden)
            self.fc2 = ComplexLinear(num_hidden, num_features)
        else:
            self.fc1 = nn.Linear(num_features, num_hidden)
            self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout1 = ComplexDropout(dropout)
        self.dropout2 = ComplexDropout(dropout)

    def forward(self, x):
        x = self.fc1(x)

        if self.complex and x.dtype in [torch.complex64, torch.complex128]:
            x = complex_non_linearity(x)
        else:
            x = F.gelu(x)
        x = self.dropout1(x)
        out = self.dropout2(self.fc2(x))
        return out


def fourier_transform(x):
    return torch.fft.rfft(x, dim=-1)


class FNetEncoderLayer(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout):
        super().__init__()
        self.ff_freq = FeedForward(d_model, expansion_factor, dropout, complex=True)
        self.ff_time = FeedForward(d_model, expansion_factor, dropout)
        # self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        n = x.shape[-2]
        x = torch.fft.rfft(x, dim=-2, n=n)
        x = self.ff_freq(x)
        # x = self.norm1(x)
        x = torch.fft.irfft(x, dim=-2, n=n)
        x = self.ff_time(x)
        out = self.norm2(x + residual)
        return out


class FNet(nn.TransformerEncoder):
    def __init__(
        self,
        d_model=256,
        expansion_factor=2,
        dropout=0.5,
        num_layers=6,
    ):
        encoder_layer = FNetEncoderLayer(d_model, expansion_factor, dropout)
        super().__init__(encoder_layer=encoder_layer, num_layers=num_layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class FNetModel(SourceModelBase):
    def __init__(
        self,
        n_freq=257,
        n_mels=64,
        expansion_factor=2,
        dropout=0.5,
        num_layers=6,
        eps=1e-6,
    ):
        super().__init__()

        self.eps = eps

        self.fnet = FNet(
            d_model=n_mels,
            expansion_factor=expansion_factor,
            dropout=dropout,
            num_layers=num_layers,
        )

        self.mel_layer = MelScale(n_stft=n_freq, n_mels=n_mels)
        self.output_mapping = nn.Linear(n_mels, n_freq)

    def forward(self, x):
        batch_shape = x.shape[:-2]
        n_freq, n_frames = x.shape[-2:]
        x = x.reshape((-1, n_freq, n_frames))

        # log-mel
        x = x.real ** 2 + x.imag ** 2
        x = self.mel_layer(x)
        x = 10.0 * torch.log10(self.eps + x)

        # transformer-like (feature should be last)
        x = x.transpose(-2, -1)
        x = self.fnet(x)

        # output mapping
        x = self.output_mapping(x)

        x = torch.sigmoid(self.eps + (1 - self.eps) * x)

        # go back to feature (freq) second
        x = x.transpose(-2, -1)

        # restore batch shape
        x = x.reshape(batch_shape + x.shape[-2:])

        return x
