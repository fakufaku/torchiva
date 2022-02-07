import torch
import torch.nn as nn


from .base import SourceModelBase
from ..linalg import mag_sq, divide


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
            layers += [
                nn.Dropout(p=dropout_p),
                GLULayer(n_hidden, n_hidden, n_sublayers=1, pool_size=pool_size),
            ]

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

        return X.reshape(batch_shape + (n_freq, n_frames))
