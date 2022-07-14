import torch
from torch import nn

from ..linalg import divide, mag_sq


class GLULayer(nn.Module):
    def __init__(
        self, n_input, n_output, n_sublayers=3, kernel_size=3, pool_size=2, eps=1e-5
    ):
        super().__init__()

        self.args = (n_input, n_output)
        self.kwargs = {
            "n_sublayers": n_sublayers,
            "kernel_size": kernel_size,
            "pool_size": pool_size,
        }

        lin_bn_layers = []
        lin_layers = []
        gate_bn_layers = []
        gate_layers = []
        pool_layers = []

        conv_type = nn.Conv1d if n_input >= n_output else nn.ConvTranspose1d
        # conv_type = nn.Conv1d

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

            pool_layers.append(
                nn.MaxPool1d(
                    kernel_size=pool_size,
                )
            )

            self.lin_layers = nn.ModuleList(lin_layers)
            self.lin_bn_layers = nn.ModuleList(lin_bn_layers)
            self.gate_layers = nn.ModuleList(gate_layers)
            self.gate_bn_layers = nn.ModuleList(gate_bn_layers)
            self.pool_layers = nn.ModuleList(pool_layers)

    def _get_args_kwargs(self):
        return self.args, self.kwargs

    def apply_constraints(self):
        pass

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
            X = pool(X.transpose(1, 2)).transpose(1, 2)

        return X


class GLUMask(nn.Module):
    def __init__(
        self,
        n_freq,
        n_bottleneck,
        pool_size=2,
        kernel_size=3,
        dropout_p=0.5,
        mag_spec=True,
        log_spec=True,
        n_sublayers=1,
    ):
        super().__init__()

        self.mag_spec = mag_spec
        self.log_spec = log_spec

        if mag_spec:
            n_inputs = n_freq
        else:
            n_inputs = 2 * n_freq

        self.layers = nn.ModuleList(
            [
                GLULayer(n_inputs, n_bottleneck, n_sublayers=1, pool_size=pool_size),
                GLULayer(
                    n_bottleneck,
                    n_bottleneck,
                    n_sublayers=n_sublayers,
                    pool_size=pool_size,
                ),
                nn.Dropout(p=dropout_p),
                GLULayer(
                    n_bottleneck,
                    n_bottleneck,
                    n_sublayers=n_sublayers,
                    pool_size=pool_size,
                ),
                nn.ConvTranspose1d(
                    in_channels=n_bottleneck,
                    out_channels=n_freq,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                ),
            ]
        )

    def apply_constraints_(self):
        pass

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

        return weights.reshape(batch_shape + (n_freq, n_frames))
