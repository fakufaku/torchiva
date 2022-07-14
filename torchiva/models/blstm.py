import torch
import torch.nn as nn
import torch.nn.functional as F
from torchaudio.transforms import MelScale

from ..linalg import mag_sq


class BLSTMMask(nn.Module):
    def __init__(
        self,
        n_freq,
        n_bottleneck,
        n_hidden=256,
        dropout_p=0.5,
        n_sources=2,
        mag_spec=True,
        log_spec=False,
    ):
        super().__init__()

        self.mag_spec = mag_spec
        self.log_spec = log_spec

        if mag_spec:
            n_inputs = n_bottleneck
        else:
            n_inputs = 2 * n_bottleneck

        self.mel_layer = MelScale(n_stft=n_freq, n_mels=n_bottleneck)

        self.blstm = nn.LSTM(
            n_inputs,
            hidden_size=n_hidden,
            num_layers=3,
            dropout=dropout_p,
            bidirectional=True,
        )

        self.output_layer = nn.Conv1d(
            in_channels=n_hidden * 2,  # *2 is because LSTM is bi-directional
            out_channels=n_freq,
            kernel_size=1,
            padding=0,
        )

    def apply_constraints_(self):
        pass

    def forward(self, X):

        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        X = X.reshape((-1, n_freq, n_frames))

        # take absolute value, also makes complex value real
        # manual implementation of complex magnitude
        if self.mag_spec:
            X = mag_sq(X)
            X = self.mel_layer(X)

            if self.log_spec:
                X = torch.log(X + 1e-7)
        else:
            # reduce to mel-spectrogram
            X = torch.view_as_real(X)
            Xr = self.mel_layer(X[..., 0])
            Xi = self.mel_layer(X[..., 1])

            # concatenate real and imaginary
            X = torch.cat((Xr, Xi), dim=-2)

        # apply all the shared layers
        X = X.permute([2, 0, 1])  # -> (n_frames, n_batch, n_freq)
        X = self.blstm(X)[0]  # -> (n_frames, n_batch, n_hidden * 2)
        X = X.permute([1, 2, 0])  # -> (n_batch, n_hidden * 2, n_frames)

        X = self.output_layer(X)

        # transform to weight by applying the sigmoid
        weights = torch.sigmoid(X)

        # add a small positive offset to the weights
        weights = weights * (1 - 1e-5) + 1e-5

        return weights.reshape(batch_shape + (n_freq, n_frames))


class MaskMVDRSupport(nn.Module):
    """
    This module procures the masks needed for the MVDR beamformer described in
    C. Boeddeker et al., "CONVOLUTIVE TRANSFER FUNCTION INVARIANT SDR TRAINING
    CRITERIA FOR MULTI-CHANNEL REVERBERANT SPEECH SEPARATION", Proc. ICASSP
    2021.
    """

    def __init__(
        self,
        *args,
        n_src=2,
        n_masks=3,
        n_input=512,
        n_hidden=600,
        dropout_p=0.5,
        n_layers=3,
        eps=1e-3
    ):
        super().__init__()

        self.eps = eps
        self.n_src = n_src
        self.n_masks = n_masks

        self.blstm = nn.LSTM(
            input_size=n_input,
            hidden_size=n_hidden,
            num_layers=n_layers,
            dropout=dropout_p,
            bidirectional=True,
        )
        self.ff1 = torch.nn.Linear(2 * n_hidden, 2 * n_hidden)
        self.ff2 = torch.nn.Linear(2 * n_hidden, n_masks * n_src * n_input)

    def forward(self, X):
        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        # flatten the batch
        X = X.reshape((-1, n_freq, n_frames))

        # input features
        X = torch.log(1.0 + X.abs())

        # BLSTM
        X = X.permute([2, 0, 1])  # -> (n_frames, n_batch, n_freq)
        X = self.blstm(X)[0]  # -> (n_frames, n_batch, n_hidden * 2)

        # linear 1
        X = F.relu(self.ff1(X))

        # linear 2
        X = torch.sigmoid(self.ff2(X))

        # re-order
        X = X.permute([1, 2, 0])  # -> (n_batch, n_freq, n_frames)

        X = (1 - self.eps) * X + self.eps

        # restore batch size
        X = X.reshape(batch_shape + X.shape[-2:])

        X = X.reshape(X.shape[:-2] + (self.n_src, self.n_masks, -1) + X.shape[-1:])

        return X
