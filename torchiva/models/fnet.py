import math
import torch
from torch import nn
from torch.nn import functional as F

from torchaudio.transforms import MelScale
from .base import SourceModelBase
from .complex_fnet import ComplexFNet


class FeedForward(nn.Module):
    def __init__(self, num_features, expansion_factor, dropout):
        super().__init__()
        num_hidden = expansion_factor * num_features
        self.fc1 = nn.Linear(num_features, num_hidden)
        self.fc2 = nn.Linear(num_hidden, num_features)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x):
        x = self.dropout1(F.gelu(self.fc1(x)))
        out = self.dropout2(self.fc2(x))
        return out


def fourier_transform(x):
    return torch.fft.fft2(x, dim=(-1, -2)).real


class FNetEncoderLayer(nn.Module):
    def __init__(self, d_model, expansion_factor, dropout):
        super().__init__()
        self.ff = FeedForward(d_model, expansion_factor, dropout)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x):
        residual = x
        x = fourier_transform(x)
        x = self.norm1(x + residual)
        residual = x
        x = self.ff(x)
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
        use_complex=False,
        eps=1e-6,
    ):
        super().__init__()

        self.eps = eps

        if use_complex:
            assert num_layers % 2 == 0
            num_layers = math.ceil(num_layers / 2)
            self.fnet = ComplexFNet(
                d_model=n_mels,
                expansion_factor=expansion_factor,
                dropout=dropout,
                num_layers=num_layers,
            )
        else:
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
