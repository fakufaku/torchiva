import torch
import torch.nn as nn
import torchaudio

from ..linalg import divide, mag_sq
from .base import SourceModelBase


class GLULayer(SourceModelBase):
    def __init__(
        self, n_input, n_output, n_sublayers=3, kernel_size=3, pool_size=2, eps=1e-5
    ):
        super().__init__()

        lin_bn_layers = []
        lin_layers = []
        gate_bn_layers = []
        gate_layers = []
        pool_layers = []

        conv_type = nn.Conv1d if n_input >= n_output else nn.ConvTranspose1d

        for n in range(n_sublayers):
            n_out = n_output if n == n_sublayers - 1 else n_input

            lin_layers.append(
                conv_type(
                    in_channels=n_input,
                    out_channels=pool_size * n_out,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            lin_bn_layers.append(nn.BatchNorm1d(pool_size * n_out))

            gate_layers.append(
                conv_type(
                    in_channels=n_input,
                    out_channels=pool_size * n_out,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            gate_bn_layers.append(nn.BatchNorm1d(pool_size * n_out))

            pool_layers.append(nn.MaxPool1d(kernel_size=pool_size,))

        self.lin_layers = nn.ModuleList(lin_layers)
        self.lin_bn_layers = nn.ModuleList(lin_bn_layers)
        self.gate_layers = nn.ModuleList(gate_layers)
        self.gate_bn_layers = nn.ModuleList(gate_bn_layers)
        self.pool_layers = nn.ModuleList(pool_layers)

    def forward(self, X):

        for lin, lin_bn, gate, gate_bn, pool in zip(
            self.lin_layers,
            self.lin_bn_layers,
            self.gate_layers,
            self.gate_bn_layers,
            self.pool_layers,
        ):
            G = gate(X)
            G = gate_bn(G)
            G = torch.sigmoid(G)
            X = lin(X)
            X = lin_bn(X)
            X = G * X
            X = pool(X.transpose(-1, -2)).transpose(-1, -2)

        return X


class GLUMask(SourceModelBase):
    def __init__(
        self,
        n_input,
        n_output,
        n_hidden,
        n_layers=3,
        pool_size=2,
        kernel_size=3,
        dropout_p=0.5,
        eps=1e-5,
        norm_time=True,
    ):
        super().__init__()

        self.norm_time = norm_time
        self.eps = eps

        layers = [GLULayer(n_input, n_hidden, n_sublayers=1, pool_size=pool_size)]

        for n in range(1, n_layers):
            if n == n_layers - 1:
                layers.append(nn.Dropout(p=dropout_p))
            layers.append(
                GLULayer(n_hidden, n_hidden, n_sublayers=1, pool_size=pool_size)
            )

        layers.append(
            nn.ConvTranspose1d(
                in_channels=n_hidden,
                out_channels=n_output,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )

        self.layers = nn.Sequential(*layers)

    def forward(self, X):

        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        X = X.reshape((-1, n_freq, n_frames))
        X_pwr = mag_sq(X)

        # we want to normalize the scale of the input signal
        g = torch.clamp(torch.mean(X_pwr, dim=(-2, -1), keepdim=True), min=self.eps)
        X = divide(X_pwr, g)

        # log-scale
        X = torch.log(1.0 + X)

        # apply all the layers
        X = self.layers(X)

        # transform to weight by applying the sigmoid
        X = torch.sigmoid(X)

        # add a small positive offset to the weights
        X = X * (1 - self.eps) + self.eps

        if self.norm_time:
            X = X / torch.sum(X, dim=-1, keepdim=True)

        X = X.reshape(batch_shape + X.shape[-2:])

        X = torch.broadcast_to(X, batch_shape + (n_freq, n_frames))

        return X


class GLUMask2(SourceModelBase):
    def __init__(
        self,
        n_freq,
        n_bottleneck,
        n_output=None,
        pool_size=2,
        kernel_size=3,
        dropout_p=0.5,
        mag_spec=True,
        log_spec=True,
        norm_time=False,
    ):
        super().__init__()

        self.mag_spec = mag_spec
        self.log_spec = log_spec
        self.norm_time = norm_time

        self.use_output_pulling = False

        if n_output is None:
            n_output = n_freq

        if mag_spec:
            n_inputs = n_freq
        else:
            n_inputs = 2 * n_freq

        self.layers = nn.ModuleList(
            [
                GLULayer(n_inputs, n_bottleneck, n_sublayers=1, pool_size=pool_size),
                GLULayer(
                    n_bottleneck, n_bottleneck, n_sublayers=1, pool_size=pool_size
                ),
                nn.Dropout(p=dropout_p),
                GLULayer(
                    n_bottleneck, n_bottleneck, n_sublayers=1, pool_size=pool_size
                ),
                nn.ConvTranspose1d(
                    in_channels=n_bottleneck,
                    out_channels=n_output,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
            ]
        )

    def enable_output_pooling(self, kernel_size=32):
        self.use_output_pulling = True
        self.kernel_size = kernel_size

    def disable_output_pooling(self):
        self.use_output_pulling = False

    def pool_mask(self, mask):

        n_freq = mask.shape[-2]

        # pooling acts on freq dim.
        mask = mask.transpose(-2, -1)

        # linearlize batch size
        batch_shape = mask.shape[:-2]
        mask = mask.reshape((-1,) + mask.shape[-2:])

        # *MIN* pooling
        mask = -torch.nn.functional.max_pool1d(
            -mask, self.kernel_size, self.kernel_size, ceil_mode=True
        )

        # repeat elements to match original size
        mask = torch.stack([mask] * self.kernel_size, dim=-1)
        mask = mask.reshape(mask.shape[:-2] + (mask.shape[-2] * mask.shape[-1],))
        mask = mask[..., :n_freq]

        # reshape batch
        mask = mask.reshape(batch_shape + mask.shape[-2:])

        # re-order dimension to original
        mask = mask.transpose(-2, -1)

        return mask

    def forward(self, X):

        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        X = X.reshape((-1, n_freq, n_frames))
        X_pwr = mag_sq(X)

        # we want to normalize the scale of the input signal
        g = torch.clamp(torch.mean(X_pwr, dim=(-2, -1), keepdim=True), min=1e-5)

        # take absolute value, also makes complex value real
        # manual implementation of complex magnitude
        if self.mag_spec:
            X = divide(X_pwr, g)
        else:
            X = divide(X, torch.sqrt(g))
            X = torch.view_as_real(X)
            X = torch.cat((X[..., 0], X[..., 1]), dim=-2)

        # work with something less prone to explode
        if self.log_spec:
            X = torch.abs(X)
            weights = torch.log10(X + 1e-7)
        else:
            weights = X

        # apply all the layers
        for idx, layer in enumerate(self.layers):
            weights = layer(weights)

        # transform to weight by applying the sigmoid
        weights = torch.sigmoid(weights)

        # add a small positive offset to the weights
        weights = weights * (1 - 1e-5) + 1e-5

        if self.use_output_pulling:
            weights = self.pool_mask(weights)

        if self.norm_time:
            weights = weights / torch.sum(weights, dim=-1, keepdim=True)

        weights = weights.reshape(batch_shape + weights.shape[-2:])
        weights = torch.broadcast_to(weights, weights.shape[:-2] + (n_freq, n_frames))


        return weights


class MelGLUMask(SourceModelBase):
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
        fbank = torchaudio.functional.melscale_fbanks(
            n_freqs=n_freq,
            f_min=0.0,
            f_max=sample_rate // 2,
            n_mels=n_mels,
            sample_rate=sample_rate,
        )
        self.register_buffer("fbank", fbank)

        # pseudo-inverse
        CC = fbank.transpose(-2, -1) @ fbank
        CC[0, 0] = 1.0
        inv_fbank = torch.linalg.inv(CC) @ fbank.transpose(-2, -1)
        self.register_buffer("inv_fbank", inv_fbank)

        if kernel_size % 2 != 1:
            raise ValueError("The kernel size should be odd")

        self.conv = torch.nn.Conv1d(
            n_mels, pool_size * n_hidden, kernel_size, padding=kernel_size // 2
        )
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


class MelGLUMask(SourceModelBase):
    def __init__(
        self,
        n_freq,
        n_bottleneck,
        pool_size=2,
        kernel_size=3,
        dropout_p=0.5,
        sample_rate=16000,
        norm_time=False,
        eps=1e-5,
    ):
        super().__init__()

        self.norm_time = norm_time
        self.eps = eps

        # mel-scale filter bank
        fbank = torchaudio.functional.melscale_fbanks(
            n_freqs=n_freq,
            f_min=0.0,
            f_max=sample_rate // 2,
            n_mels=n_bottleneck,
            sample_rate=sample_rate,
        )
        self.register_buffer("fbank", fbank)

        # pseudo-inverse
        CC = fbank.transpose(-2, -1) @ fbank
        CC[0, 0] = 1.0
        inv_fbank = torch.linalg.inv(CC) @ fbank.transpose(-2, -1)
        self.register_buffer("inv_fbank", inv_fbank)

        # middle layers
        self.layers = nn.Sequential(
            GLULayer(n_bottleneck, n_bottleneck, n_sublayers=1, pool_size=pool_size),
            GLULayer(n_bottleneck, n_bottleneck, n_sublayers=1, pool_size=pool_size),
            nn.Dropout(p=dropout_p),
            GLULayer(n_bottleneck, n_bottleneck, n_sublayers=1, pool_size=pool_size),
        )

    def forward(self, X):

        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        X = X.reshape((-1, n_freq, n_frames))
        X_pwr = mag_sq(X)

        # we want to normalize the scale of the input signal
        g = torch.clamp(torch.mean(X_pwr, dim=(-2, -1), keepdim=True), min=1e-5)

        X = divide(X_pwr, g)

        # apply mel-filter
        X = (X.transpose(-2, -1) @ self.fbank).transpose(-2, -1)

        # log domain
        X = torch.log10(X + 1e-7)

        # apply all the layers
        X = self.layers(X)

        # make positive
        X = X ** 2

        # "inverse" mel-filter
        X = (X.transpose(-2, -1) @ self.inv_fbank).transpose(-2, -1)

        # transform to weight by applying the sigmoid
        X = torch.sigmoid(X)

        # add a small positive offset to the weights
        X = X * (1 - self.eps) + self.eps

        if self.norm_time:
            X = X / torch.sum(X, dim=-1, keepdim=True)

        X = X.reshape(batch_shape + X.shape[-2:])

        return X
