import torch
from torch import nn

from .base import SourceModelBase
from torchaudio.transforms import MelScale
import torchaudio


class SimpleModel(SourceModelBase):
    def __init__(
        self,
        n_freq=257,
        n_mels=64,
        eps=1e-6,
    ):
        super().__init__()

        self.eps = eps

        self.mel_layer = MelScale(n_stft=n_freq, n_mels=n_mels)
        self.output_mapping = nn.Linear(n_mels, n_freq)

    def forward(self, x):
        batch_shape = x.shape[:-2]
        n_freq, n_frames = x.shape[-2:]
        x = x.reshape((-1, n_freq, n_frames))

        # log-mel
        x = x.abs() ** 2
        x = self.mel_layer(x)
        x = 10.0 * torch.log10(self.eps + x)

        x = x.transpose(-2, -1)

        # output mapping
        x = self.output_mapping(x)

        x = torch.sigmoid(self.eps + (1 - self.eps) * x)

        # go back to feature (freq) second
        x = x.transpose(-2, -1)

        # restore batch shape
        x = x.reshape(batch_shape + x.shape[-2:])

        return x

class SimpleModel2(SourceModelBase):
    def __init__(
        self,
        n_freq=257,
        n_mels=64,
        n_hidden=None,
        kernel_size=15,
        pool_size=7,
        dropout_p=0.1,
        sample_rate=16000,
        eps=1e-5,
    ):
        super().__init__()

        self.eps = eps

        if n_hidden is None:
            n_hidden = n_mels

        # mel-scale filter bank
        fbank = torchaudio.functional.melscale_fbanks(n_freqs=n_freq, f_min=0.0, f_max=sample_rate // 2, n_mels=n_mels, sample_rate=sample_rate)
        self.register_buffer("fbank", fbank)

        # pseudo-inverse
        CC = fbank.transpose(-2, -1) @ fbank
        CC[0, 0] = 1.0
        inv_fbank = torch.linalg.inv(CC) @ fbank.transpose(-2, -1)
        self.register_buffer("inv_fbank", inv_fbank)

        if kernel_size % 2 != 1:
            raise ValueError("The kernel size should be odd")

        self.conv = torch.nn.Conv1d(n_mels, pool_size * n_hidden, kernel_size, padding=kernel_size // 2)
        self.pool = torch.nn.MaxPool1d(pool_size)
        self.drop = torch.nn.Dropout(dropout_p)
        self.proj = torch.nn.Conv1d(n_hidden, n_mels, 1)

    def forward(self, x):
        batch_shape = x.shape[:-2]
        n_freq, n_frames = x.shape[-2:]
        x = x.reshape((-1, n_freq, n_frames))

        # log-mel
        x = x.abs() ** 2
        x = (x.transpose(-2, -1) @ self.fbank).transpose(-2, -1)
        x = 10.0 * torch.log10(self.eps + x)

        x = torch.relu(self.conv(x))
        x = self.pool(x.transpose(-2, -1)).transpose(-2, -1)
        x = self.proj(self.drop(x))

        # output mapping
        x = (x.transpose(-2, -1) @ self.inv_fbank).transpose(-2, -1)

        x = self.eps + (1 - self.eps) * torch.sigmoid(x)

        # restore batch shape
        x = x.reshape(batch_shape + x.shape[-2:])

        return x
