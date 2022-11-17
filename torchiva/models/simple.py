import torch
from torch import nn
from torchaudio.transforms import MelScale


class SimpleModel(nn.Module):
    def __init__(
        self,
        n_freq=257,
        n_mels=64,
        n_hidden=None,
        eps=1e-6,
    ):
        super().__init__()

        self.eps = eps

        if n_hidden is None:
            n_hidden = n_mels

        self.mel_layer = MelScale(n_stft=n_freq, n_mels=n_mels)
        self.hidden = nn.Linear(n_mels, n_hidden)
        self.output_mapping = nn.Linear(n_hidden, n_freq)

    def forward(self, x):
        batch_shape = x.shape[:-2]
        n_freq, n_frames = x.shape[-2:]
        x = x.reshape((-1, n_freq, n_frames))

        # log-mel
        x = x.abs() ** 2
        x = self.mel_layer(x)
        x = 10.0 * torch.log10(self.eps + x)

        x = x.transpose(-2, -1)

        # hidden layer
        x = self.hidden(x)

        # output mapping
        x = self.output_mapping(x)

        x = torch.sigmoid(self.eps + (1 - self.eps) * x)

        # go back to feature (freq) second
        x = x.transpose(-2, -1)

        # restore batch shape
        x = x.reshape(batch_shape + x.shape[-2:])

        return x
