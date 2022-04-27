# Copyright (c) 2022 Robin Scheibler, Kohei Saijo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


import torch as pt
import torch.nn as nn
import torch.nn.functional as F
from torchaudio import functional as F_audio
from torchaudio.transforms import InverseMelScale, MelScale

import torchiva as bss


def get_model(name: str, kwargs: dict):
    """
    Get a model by its name

    Parameters
    ----------
    name: str
        Name of the model class
    kwargs: dict
        A dict containing all the arguments to the model
    """

    # we get a reference to all the objects in the current module
    d = globals()
    return d[name](**kwargs)


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
        self.ff1 = pt.nn.Linear(2 * n_hidden, 2 * n_hidden)
        self.ff2 = pt.nn.Linear(2 * n_hidden, n_masks * n_src * n_input)

    def forward(self, X):
        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        # flatten the batch
        X = X.reshape((-1, n_freq, n_frames))

        # input features
        X = pt.log(1.0 + X.abs())

        # BLSTM
        X = X.permute([2, 0, 1])  # -> (n_frames, n_batch, n_freq)
        X = self.blstm(X)[0]  # -> (n_frames, n_batch, n_hidden * 2)

        # linear 1
        X = F.relu(self.ff1(X))

        # linear 2
        X = pt.sigmoid(self.ff2(X))

        # re-order
        X = X.permute([1, 2, 0])  # -> (n_batch, n_freq, n_frames)

        X = (1 - self.eps) * X + self.eps

        # restore batch size
        X = X.reshape(batch_shape + X.shape[-2:])

        X = X.reshape(X.shape[:-2] + (self.n_src, self.n_masks, -1) + X.shape[-1:])

        return X


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

            pool_layers.append(nn.MaxPool1d(kernel_size=pool_size,))

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
            G = pt.sigmoid(G)
            X = lin(X)
            X = lin_bn(X)
            X = G * X
            X = pool(X.transpose(1, 2)).transpose(1, 2)

        return X


class GLULogLayer(nn.Module):
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

            pool_layers.append(nn.MaxPool1d(kernel_size=pool_size,))

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
            G = gate(pt.log(F.relu(X) + 1e-5))
            G = gate_bn(G)
            G = pt.sigmoid(G)
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
        laplace_reg=False,
        laplace_ratio=0.99,
        mag_spec=True,
        log_spec=True,
        n_sublayers=1,
    ):
        super().__init__()

        self.args = (n_freq, n_bottleneck)
        self.kwargs = {
            "pool_size": pool_size,
            "kernel_size": kernel_size,
            "dropout_p": dropout_p,
            "laplace_reg": laplace_reg,
        }

        self.laplace_reg = laplace_reg
        self.w = laplace_ratio

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
                    n_bottleneck, n_bottleneck, n_sublayers=n_sublayers, pool_size=pool_size,
                ),
                nn.Dropout(p=dropout_p),
                GLULayer(
                    n_bottleneck, n_bottleneck, n_sublayers=n_sublayers, pool_size=pool_size,
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
        X_pwr = bss.linalg.mag_sq(X)

        # we want to normalize the scale of the input signal
        g = pt.clamp(pt.mean(X_pwr, dim=(-2, -1), keepdim=True), min=1e-5)

        # take absolute value, also makes complex value real
        # manual implementation of complex magnitude
        if self.mag_spec:
            X = bss.linalg.divide(X_pwr, g)
        else:
            X = bss.linalg.divide(X, pt.sqrt(g))
            X = pt.view_as_real(X)
            X = pt.cat((X[..., 0], X[..., 1]), dim=-2)

        # work with something less prone to explode
        if self.log_spec:
            X = pt.abs(X)
            weights = pt.log10(X + 1e-7)
        else:
            weights = X

        # apply all the layers
        for idx, layer in enumerate(self.layers):
            weights = layer(weights)

        if self.laplace_reg:
            # compute the Laplace activations, i.e. root mean power over frequencies
            X = pt.sqrt(X.mean(dim=-2, keepdim=True) + 1e-5)
            X, weights = pt.broadcast_tensors(X, weights)
            # modified sigmoid including DNN output and Laplace regularizer
            # we use a power of 10 because we used log10 above
            # also, this makes it easy to limit the maximum exponent to a power of 10,
            # e.g. 10^10, here
            X = self.w * X + (1.0 - self.w) * pt.pow(10.0, pt.clamp(-weights, max=10))
            weights = pt.reciprocal(1.0 + X)
        else:
            # transform to weight by applying the sigmoid
            weights = pt.sigmoid(weights)

        # add a small positive offset to the weights
        weights = weights * (1 - 1e-5) + 1e-5

        return weights.reshape(batch_shape + (n_freq, n_frames))



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
            X = bss.linalg.mag_sq(X)
            X = self.mel_layer(X)

            if self.log_spec:
                X = pt.log(X + 1e-7)
        else:
            # reduce to mel-spectrogram
            X = pt.view_as_real(X)
            Xr = self.mel_layer(X[..., 0])
            Xi = self.mel_layer(X[..., 1])

            # concatenate real and imaginary
            X = pt.cat((Xr, Xi), dim=-2)

        # apply all the shared layers
        X = X.permute([2, 0, 1])  # -> (n_frames, n_batch, n_freq)
        X = self.blstm(X)[0]  # -> (n_frames, n_batch, n_hidden * 2)
        X = X.permute([1, 2, 0])  # -> (n_batch, n_hidden * 2, n_frames)

        X = self.output_layer(X)

        # transform to weight by applying the sigmoid
        weights = pt.sigmoid(X)

        # add a small positive offset to the weights
        weights = weights * (1 - 1e-5) + 1e-5

        return weights.reshape(batch_shape + (n_freq, n_frames))
