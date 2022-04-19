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


from abc import ABC, abstractmethod

import yaml
import torch as pt
import torch.nn as nn
import torch.nn.functional as F
import torchaudio
from torchaudio import functional as F_audio
from torchaudio.transforms import InverseMelScale, MelScale

import torchiva as bss
from torchiva.models.base import SourceModelBase
from torchiva.models import LaplaceModel, GaussModel


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


class BaseModule(nn.Module):
    def __init__(self):
        super().__init__()

    def get_state(self):
        args, kwargs = self._get_args_kwargs()
        return {
            "name": self.__name__,
            "args": args,
            "kwargs": kwargs,
            "state_dict": self.state_dict(),
        }

    @abstractmethod
    def _get_args_kwargs(self):
        raise NotImplementedError

    @abstractmethod
    def apply_constraints(self):
        raise NotImplementedError


class BLSTM_FC(SourceModelBase, nn.Module):
    def __init__(
        self,
        n_input=2049,
        n_lstm_hidden=64,
        n_fc_hidden=256,
        n_output=1,
        n_layers=1,
        dropout_p=0.5,
        eps=1e-5,
    ):

        super().__init__()

        self.fc1 = nn.Linear(n_input, n_fc_hidden)
        self.bn1 = nn.BatchNorm1d(n_fc_hidden)
        self.blstm = nn.LSTM(
            input_size=n_fc_hidden,
            hidden_size=n_lstm_hidden,
            num_layers=n_layers,
            dropout=dropout_p,
            bidirectional=True,
            batch_first=True,
        )
        self.fc2 = nn.Linear(n_lstm_hidden*2, n_output)
        self.eps = eps
        self.n_fc_hidden = n_fc_hidden

    def forward(self, X):
        
        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        # flatten the batch
        X = X.reshape((-1, n_freq, n_frames))

        # input features
        #X = pt.log(1.0 + X.abs())
        X = pt.log(abs(X) + 1e-5)
        X = X.permute([0, 2, 1])  # -> (n_batch*n_chan, n_frames, n_freq)

        Y = self.fc1(X)
        Y = pt.relu(Y.reshape(-1,self.n_fc_hidden)).reshape(-1, n_frames, self.n_fc_hidden)
        Y = self.blstm(Y)[0]
        Y = self.fc2(Y)

        v = pt.sigmoid(Y) #(n_batch*n_chan, n_frames, 1)

        v = (1 - self.eps) * v + self.eps
        v = v.reshape(batch_shape + (1, -1)) #(n_batch, n_chan, 1, n_frames)
        v = pt.tile(v,(1,1,n_freq,1))

        return v


class MaskMVDRSupport(SourceModelBase, nn.Module):
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


class FullyConvSourceModel(SourceModelBase, nn.Module): 
    def __init__(self, n_layers, kernel_size=3, eps=1e-5, mixed_laplace=True):

        super().__init__()

        self.args = (n_layers, kernel_size)
        self.kwargs = {"eps": eps, "mixed_laplace": mixed_laplace}

        layers = []
        self.norm = []
        for i, o in zip(n_layers[:-1], n_layers[1:]):
            layers.append(
                nn.Conv1d(
                    in_channels=i,
                    out_channels=o,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            self.norm.append(i * kernel_size)
        self.layers = nn.ModuleList(layers)
        self.eps = eps

        if mixed_laplace:
            self.w = nn.Parameter(pt.tensor(0.95))
        else:
            self.w = None

    def _get_args_kwargs(self):
        return self.args, self.kwargs

    def apply_constraints_(self):
        if self.w is not None:
            self.w.data.clamp_(min=0.0, max=1.0)

    def forward(self, X):

        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        X_ = X.reshape((-1, n_freq, n_frames))

        # take absolute value, also makes complex value real
        # manual implementation of complex magnitude
        Y = bss.linalg.mag_sq(X_)

        if self.w is not None:
            Y_laplace = pt.sqrt(Y.mean(dim=-2, keepdim=True) + 1e-5)

        for layer, n in zip(self.layers, self.norm):
            Y = layer(Y)
            Y = F.relu(Y)

        if self.w is not None:
            Y = (1.0 - self.w) * Y + self.w * Y_laplace

        r = Y.reshape(batch_shape + (n_freq, n_frames))

        # compute global scale
        # we want to make sure the scale does not explode
        g = pt.mean(r, dim=(-2, -1))

        # we take the inverse of the weights
        r_inv = g[..., None, None] / pt.clamp(r, min=1e-5)

        # restore batch shape
        return r_inv


class FullyConvSourceModel2(SourceModelBase, nn.Module):
    def __init__(self, n_layers, kernel_size=3, eps=1e-5, mixed_laplace=True):

        super().__init__()

        self.args = (n_layers, kernel_size)
        self.kwargs = {"eps": eps, "mixed_laplace": mixed_laplace}

        layers = []
        self.norm = []
        for i, o in zip(n_layers[:-1], n_layers[1:]):
            layers.append(
                nn.Conv1d(
                    in_channels=i,
                    out_channels=o,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            self.norm.append(i * kernel_size)
        self.layers = nn.ModuleList(layers)
        self.eps = eps
        self.mixed_laplace = mixed_laplace

    def _get_args_kwargs(self):
        return self.args, self.kwargs

    def apply_constraints_(self):
        pass

    def forward(self, X):

        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        X_ = X.reshape((-1, n_freq, n_frames))

        # take absolute value, also makes complex value real
        # manual implementation of complex magnitude
        Y = bss.linalg.mag_sq(X_)

        if self.mixed_laplace:
            Y_laplace = pt.sqrt(Y.mean(dim=-2, keepdim=True) + 1e-5)

        for layer, n in zip(self.layers, self.norm):
            Y = layer(Y)
            Y = F.relu(Y)

        if self.mixed_laplace:
            Y = Y + Y_laplace

        # restore batch shape
        r = Y.reshape(batch_shape + (n_freq, n_frames))

        # compute global scale
        # we want to make sure the scale does not explode
        g = pt.mean(r, dim=(-2, -1))

        # we take the inverse of the weights
        r_inv = g[..., None, None] / pt.clamp(r, min=1e-5)

        # restore batch shape
        return r_inv


class MelConvSourceModel(SourceModelBase, nn.Module):
    def __init__(self, n_freq, n_mels, n_layers, kernel_size=3, eps=1e-5):

        super().__init__()

        self.args = (n_freq, n_mels, n_layers)
        self.kwargs = {"kernel_size": kernel_size, "eps": eps}

        self.mel_layer = MelScale(n_stft=n_freq, n_mels=n_mels)

        fb = F_audio.create_fb_matrix(
            n_freq,
            self.mel_layer.f_min,
            self.mel_layer.f_max,
            self.mel_layer.n_mels,
            self.mel_layer.sample_rate,
        )
        self.register_buffer("fb", fb)

        self.w = nn.Parameter(pt.tensor(0.95))

        n_layers = [n_mels] + n_layers + [n_mels]

        layers = []
        self.norm = []
        for i, o in zip(n_layers[:-1], n_layers[1:]):
            layers.append(
                nn.Conv1d(
                    in_channels=i,
                    out_channels=o,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            self.norm.append(i * kernel_size)
        self.layers = nn.ModuleList(layers)
        self.eps = eps

    def _get_args_kwargs(self):
        return self.args, self.kwargs

    def apply_constraints_(self):
        self.w.data.clamp_(min=0.0, max=1.0)

    def forward(self, X):

        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        X_ = X.reshape((-1, n_freq, n_frames))

        # take absolute value, also makes complex value real
        # manual implementation of complex magnitude
        Y = bss.linalg.mag_sq(X_)

        Y2 = pt.sqrt(Y.mean(dim=-2, keepdim=True))

        Y = self.mel_layer(Y)

        for layer, n in zip(self.layers, self.norm):
            Y = layer(Y)
            Y = F.relu(Y)

        # inverse mel filter bank (transpose)
        Y = self.fb[None, ...].matmul(Y)

        Y = 1.0 / pt.clamp(self.w * Y2 + (1 - self.w) * Y, min=self.eps)

        # restore batch shape
        r = Y.reshape(batch_shape + (n_freq, n_frames))

        # compute global scale
        # we want to make sure the scale does not explode
        g = pt.mean(r, dim=(-2, -1))

        # we take the inverse of the weights
        r_inv = g[..., None, None] / pt.clamp(r, min=1e-5)

        # restore batch shape
        return r_inv


class MiniUNetSourceModel(SourceModelBase, nn.Module):
    def __init__(
        self, n_freq, n_mels, n_layers, kernel_size=3, eps=1e-5, mixed_laplace=False
    ):

        super().__init__()

        self.args = (n_freq, n_mels, n_layers)
        self.kwargs = {
            "kernel_size": kernel_size,
            "eps": eps,
            "mixed_laplace": mixed_laplace,
        }

        self.mel_layer = MelScale(n_stft=n_freq, n_mels=n_mels)

        fb = F_audio.create_fb_matrix(
            n_freq,
            self.mel_layer.f_min,
            self.mel_layer.f_max,
            self.mel_layer.n_mels,
            self.mel_layer.sample_rate,
        )
        self.register_buffer("fb", fb)

        self.w = nn.Parameter(pt.tensor(0.95)) if mixed_laplace else None

        self.first_layer = nn.Conv1d(
            in_channels=n_freq,
            out_channels=n_mels,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.last_layer = nn.ConvTranspose1d(
            in_channels=n_mels,
            out_channels=n_freq,
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )

        n_layers = [n_mels] + n_layers

        layers_forward = []
        layers_backward = []
        for i, o in zip(n_layers[:-1], n_layers[1:]):
            l = nn.Conv1d(
                in_channels=i,
                out_channels=o,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            nn.init.uniform_(l.weight, a=0.0, b=1.0 / (kernel_size * i))
            layers_forward.append(l)

        n_layers_back = n_layers[::-1]
        for i, o in zip(n_layers_back[:-1], n_layers_back[1:]):
            l = nn.ConvTranspose1d(
                in_channels=i,
                out_channels=o,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
            nn.init.uniform_(l.weight, a=0.0, b=1.0 / (kernel_size * i))
            layers_backward.append(l)

        self.layers_forward = nn.ModuleList(layers_forward)
        self.layers_backward = nn.ModuleList(layers_backward)
        self.eps = eps

    def _get_args_kwargs(self):
        return self.args, self.kwargs

    def apply_constraints_(self):
        if self.w is not None:
            self.w.data.clamp_(min=0.05, max=1.0)

    def forward(self, X):

        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        X_ = X.reshape((-1, n_freq, n_frames))

        # take absolute value, also makes complex value real
        # manual implementation of complex magnitude
        Y = bss.linalg.mag_sq(X_)
        Y = pt.sqrt(Y + 1e-5)

        # Y = self.mel_layer(Y)
        Y = self.first_layer(Y)

        Y_in = []

        for layer in self.layers_forward:
            Y_in.append(Y)
            Y = layer(Y)
            Y = F.relu(Y)

        for layer, Y_f in zip(self.layers_backward, Y_in[::-1]):
            Y = layer(Y)
            Y = F.relu(Y)
            Y = Y + Y_f

        # inverse mel filter bank (transpose)
        # Y = self.fb[None, ...].matmul(Y)
        Y = self.last_layer(Y)

        # restore batch shape
        return Y.reshape(batch_shape + (n_freq, n_frames))


class GLULayer(nn.Module):
    def __init__(
        self, n_input, n_output, n_sublayers=3, kernel_size=3, pool_size=2, n_groups=4, eps=1e-5
    ):
        super().__init__()

        self.args = (n_input, n_output)
        self.kwargs = {
            "n_sublayers": n_sublayers,
            "kernel_size": kernel_size,
            "pool_size": pool_size,
            "n_gropus": n_groups,
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
            #lin_bn_layers.append(nn.BatchNorm1d(pool_size * n_out))
            lin_bn_layers.append(nn.GroupNorm(n_groups, pool_size * n_out))

            gate_layers.append(
                conv_type(
                    in_channels=n_input,
                    out_channels=pool_size * n_out,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
            #gate_bn_layers.append(nn.BatchNorm1d(pool_size * n_out))
            gate_bn_layers.append(nn.GroupNorm(n_groups, pool_size * n_out))

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


class SelfAttentionSourceModel(SourceModelBase, nn.Module):
    def __init__(
        self,
        n_layers,
        n_sublayers=1,
        pool_size=2,
        kernel_size=3,
        eps=1e-5,
        mixed_laplace=False,
    ):
        super().__init__()

        self.args = (n_layers,)
        self.kwargs = {
            "n_sublayers": n_sublayers,
            "pool_size": pool_size,
            "kernel_size": kernel_size,
            "eps": eps,
            "mixed_laplace": mixed_laplace,
        }

        self.eps = eps
        self.mixed_laplace = mixed_laplace

        layers = []
        for i, o in zip(n_layers[:-1], n_layers[1:]):
            if i >= o:
                layers.append(
                    GLULayer(i, o, n_sublayers=n_sublayers, pool_size=pool_size)
                )
            elif i < o:
                layers.append(
                    nn.ConvTranspose1d(
                        in_channels=i,
                        out_channels=o,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    )
                )
        self.layers = nn.ModuleList(layers)

    def apply_constraints_(self):
        pass

    def forward(self, X):

        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        X_ = X.reshape((-1, n_freq, n_frames))

        # take absolute value, also makes complex value real
        # manual implementation of complex magnitude
        Y = bss.linalg.mag_sq(X_)

        if self.mixed_laplace:
            Y_laplace = pt.sqrt(Y.mean(dim=-2, keepdim=True) + 1e-5)

        Y = pt.sqrt(Y + 1e-5)

        for idx, layer in enumerate(self.layers):
            Y = layer(Y)
            if idx > 0:
                Y = F.relu(Y)

        if self.mixed_laplace:
            Y = Y_laplace + Y

        # compute global scale
        # we want to make sure the scale does not explode
        g = pt.mean(Y, dim=(-2, -1))

        # we take the inverse of the weights
        Y = g[..., None, None] / pt.clamp(Y, min=self.eps)

        return Y.reshape(batch_shape + (n_freq, n_frames))


class SelfAttentionSourceModel2(SourceModelBase, nn.Module):
    def __init__(
        self,
        n_layers,
        n_sublayers=1,
        pool_size=2,
        kernel_size=3,
        eps=1e-5,
        mixed_laplace=False,
    ):
        super().__init__()

        self.args = (n_layers,)
        self.kwargs = {
            "n_sublayers": n_sublayers,
            "pool_size": pool_size,
            "kernel_size": kernel_size,
            "eps": eps,
            "mixed_laplace": mixed_laplace,
        }

        self.eps = eps
        self.mixed_laplace = mixed_laplace

        layers = []
        for i, o in zip(n_layers[:-1], n_layers[1:]):
            if i >= o:
                layers.append(
                    GLULayer(i, o, n_sublayers=n_sublayers, pool_size=pool_size)
                )
            elif i < o:
                layers.append(
                    nn.ConvTranspose1d(
                        in_channels=i,
                        out_channels=o,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    )
                )
        self.layers = nn.ModuleList(layers)

    def apply_constraints_(self):
        pass

    def forward(self, X):

        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        X_ = X.reshape((-1, n_freq, n_frames))

        # take absolute value, also makes complex value real
        # manual implementation of complex magnitude
        Y = bss.linalg.mag_sq(X_)

        if self.mixed_laplace:
            Y_laplace = Y.mean(dim=-2, keepdim=True)

        for idx, layer in enumerate(self.layers):
            Y = layer(Y)

        Y = F.relu(Y)

        if self.mixed_laplace:
            Y = Y_laplace + Y

        # we take the inverse of the weights
        Y = 1.0 / pt.clamp(Y, min=1e-5)

        return Y.reshape(batch_shape + (n_freq, n_frames))


"""
Implementation of a sort of UNet
"""


class UNetDownSamplingBlock(nn.Module):
    def __init__(self, channels, n_sublayers=3, kernel_size=3, n_pool=2):
        super().__init__()

        layers = []
        for n in range(n_sublayers):
            layers.append(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
        self.pooling = nn.MaxPool1d(kernel_size=n_pool)
        self.layers = nn.ModuleList(layers)

    def forward(self, X):

        for i, layer in enumerate(self.layers[:-1]):
            X = F.relu(layer(X))
        X = self.layers[-1](X)  # no relu on last layer
        X = self.pooling(X.transpose(1, 2)).transpose(1, 2)

        return X


class UNetUpSamplingBlock(nn.Module):
    def __init__(self, channels, n_sublayers=3, kernel_size=3, n_up=2):
        super().__init__()

        layers = [
            nn.Conv1d(
                in_channels=2 * channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        ]
        for n in range(n_sublayers - 2):
            layers.append(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
        layers.append(
            nn.ConvTranspose1d(
                in_channels=channels,
                out_channels=n_up * channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )
        self.layers = nn.ModuleList(layers)

    def forward(self, X):
        for i, layer in enumerate(self.layers[:-1]):
            X = F.relu(layer(X))
        X = self.layers[-1](X)  # no relu on last layer
        return X


class UNetMask(SourceModelBase, nn.Module):
    def __init__(self, n_freq, kernel_size=3, eps=1e-5):

        super().__init__()

        self.args = (n_freq,)
        self.kwargs = {
            "kernel_size": kernel_size,
            "eps": eps,
        }

        self.dblock1 = UNetDownSamplingBlock(n_freq, n_sublayers=3, n_pool=4)
        self.dblock2 = UNetDownSamplingBlock(n_freq // 4, n_sublayers=3, n_pool=4)
        self.dblock3 = UNetDownSamplingBlock(n_freq // 16, n_sublayers=3, n_pool=2)
        self.dblock4 = UNetDownSamplingBlock(
            n_freq // 32, n_sublayers=3, n_pool=2
        )  # output is n_freq // 64

        self.mblock = UNetDownSamplingBlock(n_freq // 64, n_sublayers=3, n_pool=1)

        self.ublock4 = UNetUpSamplingBlock(n_freq // 64, n_sublayers=3, n_up=2)
        self.ublock3 = UNetUpSamplingBlock(n_freq // 32, n_sublayers=3, n_up=2)
        self.ublock2 = UNetUpSamplingBlock(n_freq // 16, n_sublayers=3, n_up=4)
        self.ublock1 = UNetUpSamplingBlock(n_freq // 4, n_sublayers=3, n_up=4)

        n_out = (n_freq // 64) * 64
        self.output_matching = nn.Conv1d(
            in_channels=n_out, out_channels=n_freq, kernel_size=1
        )

        self.eps = eps

    def _get_args_kwargs(self):
        return self.args, self.kwargs

    def apply_constraints_(self):
        pass

    def forward(self, X):

        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        Y = X.reshape((-1, n_freq, n_frames))

        # work with the spectrogram
        Y = bss.linalg.mag_sq(Y)

        # downsampling blocks
        Y1 = self.dblock1(Y)
        Y2 = self.dblock2(Y1)
        Y3 = self.dblock3(Y2)
        Y4 = self.dblock4(Y3)

        # center block
        Z = self.mblock(Y4)

        # upsample back to original size
        Z = self.ublock4(pt.cat([Z, Y4], dim=-2))
        Z = self.ublock3(pt.cat([Z, Y3], dim=-2))
        Z = self.ublock2(pt.cat([Z, Y2], dim=-2))
        Z = self.ublock1(pt.cat([Z, Y1], dim=-2))

        # match the output size
        Z = self.output_matching(Z)

        # finally, we use a sigmoid to restrict the output in [0, 1]
        Z = pt.sigmoid(Z)

        # and we multiply with the original spectorgram
        #Y = Z * Y

        # restore batch shape
        #return Y.reshape(batch_shape + (n_freq, n_frames))
        
        weights = Z * (1 - 1e-5) + 1e-5

        return weights.reshape(batch_shape + (n_freq, n_frames))


class UNetMask2(SourceModelBase, nn.Module):
    def __init__(self, n_freq, kernel_size=3, eps=1e-5):

        super().__init__()

        self.args = (n_freq,)
        self.kwargs = {
            "kernel_size": kernel_size,
            "eps": eps,
        }

        self.input_matching = nn.Conv1d(
            in_channels=n_freq, out_channels=n_freq // 16, kernel_size=1
        )

        self.dblock1 = UNetDownSamplingBlock(n_freq // 16, n_sublayers=3, n_pool=2)
        self.dblock2 = UNetDownSamplingBlock(n_freq // 32, n_sublayers=3, n_pool=2)
        self.dblock3 = UNetDownSamplingBlock(n_freq // 64, n_sublayers=3, n_pool=2)

        self.mblock = UNetDownSamplingBlock(n_freq // 128, n_sublayers=3, n_pool=1)

        self.ublock3 = UNetUpSamplingBlock(n_freq // 128, n_sublayers=3, n_up=2)
        self.ublock2 = UNetUpSamplingBlock(n_freq // 64, n_sublayers=3, n_up=2)
        self.ublock1 = UNetUpSamplingBlock(n_freq // 32, n_sublayers=3, n_up=2)

        self.output_matching = nn.Conv1d(
            in_channels=n_freq // 16, out_channels=n_freq, kernel_size=1
        )

        self.eps = eps

    def _get_args_kwargs(self):
        return self.args, self.kwargs

    def apply_constraints_(self):
        pass

    def forward(self, X):

        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        Y = X.reshape((-1, n_freq, n_frames))

        # work with the spectrogram
        Y = bss.linalg.mag_sq(Y)

        # apply log transform
        Y = pt.log(Y + 1e-5)

        # match the unet input size
        Y0 = self.input_matching(Y)

        # downsampling blocks
        Y1 = self.dblock1(Y0)
        Y2 = self.dblock2(Y1)
        Y3 = self.dblock3(Y2)

        # center block
        Z = self.mblock(Y3)

        # upsample back to original size
        Z = self.ublock3(pt.cat([Z, Y3], dim=-2))
        Z = self.ublock2(pt.cat([Z, Y2], dim=-2))
        Z = self.ublock1(pt.cat([Z, Y1], dim=-2))

        # match the output size
        Z = self.output_matching(Z)

        # finally, we use a sigmoid to restrict the output in [0, 1]
        Z = pt.sigmoid(Z)

        # restore batch shape
        return Z.reshape(batch_shape + (n_freq, n_frames))


class GLUMask(SourceModelBase, nn.Module):
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
        n_groups=4,
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
                GLULayer(n_inputs, n_bottleneck, n_sublayers=1, pool_size=pool_size, n_groups=n_groups),
                GLULayer(
                    n_bottleneck, n_bottleneck, n_sublayers=n_sublayers, pool_size=pool_size, n_groups=n_groups,
                ),
                nn.Dropout(p=dropout_p),
                GLULayer(
                    n_bottleneck, n_bottleneck, n_sublayers=n_sublayers, pool_size=pool_size, n_groups=n_groups,
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



class BLSTMMask(SourceModelBase, nn.Module):
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


class GLULogMask(SourceModelBase, nn.Module):
    def __init__(
        self,
        n_freq,
        n_bottleneck,
        pool_size=2,
        kernel_size=3,
        dropout_p=0.5,
        laplace_reg=False,
        laplace_ratio=0.99,
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

        self.layers = nn.ModuleList(
            [
                GLULogLayer(n_freq, n_bottleneck, n_sublayers=1, pool_size=pool_size),
                GLULogLayer(
                    n_bottleneck, n_bottleneck, n_sublayers=1, pool_size=pool_size
                ),
                nn.Dropout(p=dropout_p),
                GLULogLayer(
                    n_bottleneck, n_bottleneck, n_sublayers=1, pool_size=pool_size
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

        # take absolute value, also makes complex value real
        # manual implementation of complex magnitude
        X = bss.linalg.mag_sq(X)

        # work with something less prone to explode

        # apply all the layers
        Y = X
        for idx, layer in enumerate(self.layers):
            Y = layer(Y)

        if self.laplace_reg:
            # compute the Laplace activations, i.e. root mean power over frequencies
            X = pt.sqrt(X.mean(dim=-2, keepdim=True) + 1e-5)
            X, Y = pt.broadcast_tensors(X, Y)
            # modified sigmoid including DNN output and Laplace regularizer
            # we use a power of 10 because we used log10 above
            # also, this makes it easy to limit the maximum exponent to a power of 10,
            # e.g. 10^10, here
            Y = self.w * X + (1.0 - self.w) * Y

        Y = F.relu(Y)

        # transform to weight by applying the inverse
        weights = pt.reciprocal(1.0 + Y)

        # add a small positive offset to the weights
        weights = weights * (1 - 1e-5) + 1e-5

        return weights.reshape(batch_shape + (n_freq, n_frames))


class GLUMaskMixedLaplace(SourceModelBase, nn.Module):
    def __init__(
        self, n_layers, n_sublayers=1, pool_size=2, kernel_size=3,
    ):
        super().__init__()

        self.args = (n_layers,)
        self.kwargs = {
            "n_sublayers": n_sublayers,
            "pool_size": pool_size,
            "kernel_size": kernel_size,
            "n_sublayers": n_sublayers,
        }

        # the scale and bias weights for the laplace activation
        self.scale = nn.Parameter(pt.tensor(1.0))
        self.bias = nn.Parameter(pt.tensor(0.0))

        layers = []
        for i, o in zip(n_layers[:-1], n_layers[1:]):
            if i >= o:
                layers.append(
                    GLULayer(i, o, n_sublayers=n_sublayers, pool_size=pool_size)
                )
            elif i < o:
                layers.append(
                    nn.ConvTranspose1d(
                        in_channels=i,
                        out_channels=o,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    )
                )
        self.layers = nn.ModuleList(layers)

    def apply_constraints_(self):
        pass

    def forward(self, X):

        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        X_ = X.reshape((-1, n_freq, n_frames))

        # take absolute value, also makes complex value real
        # manual implementation of complex magnitude
        Y = bss.linalg.mag_sq(X_)
        Y_laplace = Y.mean(dim=-2, keepdim=True)

        for idx, layer in enumerate(self.layers):
            Y = layer(Y)

        # make non-negative at the end
        Y = F.relu(Y)

        # Here the idea is that we want the weight to be small if Y is large
        Y = 1.0 - pt.sigmoid(Y + self.scale * Y_laplace + self.bias)

        return Y.reshape(batch_shape + (n_freq, n_frames))


class GLUMaskSpherical(SourceModelBase, nn.Module):
    def __init__(
        self,
        n_layers,
        n_sublayers=1,
        pool_size=2,
        kernel_size=3,
        eps=1e-5,
        mixed_laplace=False,
    ):
        super().__init__()

        self.args = (n_layers,)
        self.kwargs = {
            "n_sublayers": n_sublayers,
            "pool_size": pool_size,
            "kernel_size": kernel_size,
            "eps": eps,
            "mixed_laplace": mixed_laplace,
        }

        self.eps = eps
        self.mixed_laplace = mixed_laplace

        layers = []
        for i, o in zip(n_layers[:-1], n_layers[1:]):
            if i >= o:
                layers.append(
                    GLULayer(i, o, n_sublayers=n_sublayers, pool_size=pool_size)
                )
            elif i < o:
                layers.append(
                    nn.ConvTranspose1d(
                        in_channels=i,
                        out_channels=o,
                        kernel_size=kernel_size,
                        padding=kernel_size // 2,
                    )
                )
        self.layers = nn.ModuleList(layers)

        self.last_layer = GLULayer(n_layers[-1], 1, pool_size=pool_size)

    def apply_constraints_(self):
        pass

    def forward(self, X):

        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        X_ = X.reshape((-1, n_freq, n_frames))

        # take absolute value, also makes complex value real
        # manual implementation of complex magnitude
        Y = bss.linalg.mag_sq(X_)

        for idx, layer in enumerate(self.layers):
            Y = layer(Y)

        Y = self.last_layer(Y)

        Y = pt.sigmoid(Y)

        _, Y = pt.broadcast_tensors(X_, Y)

        return Y.reshape(batch_shape + (n_freq, n_frames))


class GLUMaskMixedSpherical(SourceModelBase, nn.Module):
    def __init__(
        self,
        n_layers,
        n_sublayers=1,
        pool_size=2,
        kernel_size=3,
        eps=1e-5,
        mixed_laplace=False,
    ):
        super().__init__()

        self.args = (n_layers,)
        self.kwargs = {
            "n_sublayers": n_sublayers,
            "pool_size": pool_size,
            "kernel_size": kernel_size,
            "eps": eps,
            "mixed_laplace": mixed_laplace,
        }

        self.eps = eps
        self.mixed_laplace = mixed_laplace

        layers = []
        for i, o in zip(n_layers[:-1], n_layers[1:]):
            layers.append(GLULayer(i, o, n_sublayers=n_sublayers, pool_size=pool_size))
        self.layers = nn.ModuleList(layers)

        self.last_layer_fine = nn.ConvTranspose1d(
            in_channels=n_layers[-1],
            out_channels=n_layers[0],
            kernel_size=kernel_size,
            padding=kernel_size // 2,
        )
        self.last_layer_rough = GLULayer(n_layers[-1], 1, pool_size=pool_size)

    def apply_constraints_(self):
        pass

    def forward(self, X):

        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        X_ = X.reshape((-1, n_freq, n_frames))

        # take absolute value, also makes complex value real
        # manual implementation of complex magnitude
        Y = bss.linalg.mag_sq(X_)

        for idx, layer in enumerate(self.layers):
            Y = layer(Y)

        Y_rough = self.last_layer_rough(Y)
        Y_rough = pt.sigmoid(Y_rough)
        _, Y_rough = pt.broadcast_tensors(X_, Y_rough)

        Y_fine = self.last_layer_fine(Y)
        Y_fine = pt.sigmoid(Y_fine)

        Y = 0.5 * (Y_rough + Y_fine)

        return Y.reshape(batch_shape + (n_freq, n_frames))


"""
Implementation of a sort of UNet
"""


class UNetDownSamplingBlock(nn.Module):
    def __init__(self, channels, n_sublayers=3, kernel_size=3, n_pool=2):
        super().__init__()

        layers = []
        for n in range(n_sublayers):
            layers.append(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
        self.pooling = nn.MaxPool1d(kernel_size=n_pool)
        self.layers = nn.ModuleList(layers)

    def forward(self, X):

        for i, layer in enumerate(self.layers[:-1]):
            X = F.relu(layer(X))
        X = self.layers[-1](X)  # no relu on last layer
        X = self.pooling(X.transpose(1, 2)).transpose(1, 2)

        return X


class UNetUpSamplingBlock(nn.Module):
    def __init__(self, channels, n_sublayers=3, kernel_size=3, n_up=2):
        super().__init__()

        layers = [
            nn.Conv1d(
                in_channels=2 * channels,
                out_channels=channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        ]
        for n in range(n_sublayers - 2):
            layers.append(
                nn.Conv1d(
                    in_channels=channels,
                    out_channels=channels,
                    kernel_size=kernel_size,
                    padding=kernel_size // 2,
                )
            )
        layers.append(
            nn.ConvTranspose1d(
                in_channels=channels,
                out_channels=n_up * channels,
                kernel_size=kernel_size,
                padding=kernel_size // 2,
            )
        )
        self.layers = nn.ModuleList(layers)

    def forward(self, X):
        for i, layer in enumerate(self.layers[:-1]):
            X = F.relu(layer(X))
        X = self.layers[-1](X)  # no relu on last layer
        return X


class UNetMask(SourceModelBase, nn.Module):
    def __init__(self, n_freq, kernel_size=3, eps=1e-5):

        super().__init__()

        self.args = (n_freq,)
        self.kwargs = {
            "kernel_size": kernel_size,
            "eps": eps,
        }

        self.dblock1 = UNetDownSamplingBlock(n_freq, n_sublayers=3, n_pool=4)
        self.dblock2 = UNetDownSamplingBlock(n_freq // 4, n_sublayers=3, n_pool=4)
        self.dblock3 = UNetDownSamplingBlock(n_freq // 16, n_sublayers=3, n_pool=2)
        self.dblock4 = UNetDownSamplingBlock(
            n_freq // 32, n_sublayers=3, n_pool=2
        )  # output is n_freq // 64

        self.mblock = UNetDownSamplingBlock(n_freq // 64, n_sublayers=3, n_pool=1)

        self.ublock4 = UNetUpSamplingBlock(n_freq // 64, n_sublayers=3, n_up=2)
        self.ublock3 = UNetUpSamplingBlock(n_freq // 32, n_sublayers=3, n_up=2)
        self.ublock2 = UNetUpSamplingBlock(n_freq // 16, n_sublayers=3, n_up=4)
        self.ublock1 = UNetUpSamplingBlock(n_freq // 4, n_sublayers=3, n_up=4)

        n_out = (n_freq // 64) * 64
        self.output_matching = nn.Conv1d(
            in_channels=n_out, out_channels=n_freq, kernel_size=1
        )

        self.eps = eps

    def _get_args_kwargs(self):
        return self.args, self.kwargs

    def apply_constraints_(self):
        pass

    def forward(self, X):

        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        Y = X.reshape((-1, n_freq, n_frames))

        # work with the spectrogram
        Y = bss.linalg.mag_sq(Y)

        # downsampling blocks
        Y1 = self.dblock1(Y)
        Y2 = self.dblock2(Y1)
        Y3 = self.dblock3(Y2)
        Y4 = self.dblock4(Y3)

        # center block
        Z = self.mblock(Y4)

        # upsample back to original size
        Z = self.ublock4(pt.cat([Z, Y4], dim=-2))
        Z = self.ublock3(pt.cat([Z, Y3], dim=-2))
        Z = self.ublock2(pt.cat([Z, Y2], dim=-2))
        Z = self.ublock1(pt.cat([Z, Y1], dim=-2))

        # match the output size
        Z = self.output_matching(Z)

        # finally, we use a sigmoid to restrict the output in [0, 1]
        Z = pt.sigmoid(Z)

        # and we multiply with the original spectorgram
        Y = Z * Y

        # restore batch shape
        return Y.reshape(batch_shape + (n_freq, n_frames))


class UNetMask2(SourceModelBase, nn.Module):
    def __init__(self, n_freq, kernel_size=3, eps=1e-5):

        super().__init__()

        self.args = (n_freq,)
        self.kwargs = {
            "kernel_size": kernel_size,
            "eps": eps,
        }

        self.input_matching = nn.Conv1d(
            in_channels=n_freq, out_channels=n_freq // 16, kernel_size=1
        )

        self.dblock1 = UNetDownSamplingBlock(n_freq // 16, n_sublayers=3, n_pool=2)
        self.dblock2 = UNetDownSamplingBlock(n_freq // 32, n_sublayers=3, n_pool=2)
        self.dblock3 = UNetDownSamplingBlock(n_freq // 64, n_sublayers=3, n_pool=2)

        self.mblock = UNetDownSamplingBlock(n_freq // 128, n_sublayers=3, n_pool=1)

        self.ublock3 = UNetUpSamplingBlock(n_freq // 128, n_sublayers=3, n_up=2)
        self.ublock2 = UNetUpSamplingBlock(n_freq // 64, n_sublayers=3, n_up=2)
        self.ublock1 = UNetUpSamplingBlock(n_freq // 32, n_sublayers=3, n_up=2)

        self.output_matching = nn.Conv1d(
            in_channels=n_freq // 16, out_channels=n_freq, kernel_size=1
        )

        self.eps = eps

    def _get_args_kwargs(self):
        return self.args, self.kwargs

    def apply_constraints_(self):
        pass

    def forward(self, X):

        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        Y = X.reshape((-1, n_freq, n_frames))

        # work with the spectrogram
        Y = bss.linalg.mag_sq(Y)

        # match the unet input size
        Y0 = self.input_matching(Y)

        # downsampling blocks
        Y1 = self.dblock1(Y0)
        Y2 = self.dblock2(Y1)
        Y3 = self.dblock3(Y2)

        # center block
        Z = self.mblock(Y3)

        # upsample back to original size
        Z = self.ublock3(pt.cat([Z, Y3], dim=-2))
        Z = self.ublock2(pt.cat([Z, Y2], dim=-2))
        Z = self.ublock1(pt.cat([Z, Y1], dim=-2))

        # match the output size
        Z = self.output_matching(Z)

        # finally, we use a sigmoid to restrict the output in [0, 1]
        Z = pt.sigmoid(Z)

        # and we multiply with the original spectorgram
        Y = Z * Y

        # restore batch shape
        return Y.reshape(batch_shape + (n_freq, n_frames))


class MelGLUSourceModel(SourceModelBase, nn.Module):
    def __init__(self, n_freq, n_layers, kernel_size=3, pool_size=3, eps=1e-5):

        super().__init__()

        self.args = (n_freq, n_layers)
        self.kwargs = {"kernel_size": kernel_size, "eps": eps}

        self.mel_layer = MelScale(n_stft=n_freq, n_mels=n_layers[0])

        fb = F_audio.create_fb_matrix(
            n_freq,
            self.mel_layer.f_min,
            self.mel_layer.f_max,
            self.mel_layer.n_mels,
            self.mel_layer.sample_rate,
        )
        self.register_buffer("fb", fb)

        layers = []
        for i, o in zip(n_layers[:-1], n_layers[1:]):
            layers.append(GLULayer(i, o, kernel_size=3, pool_size=2, n_sublayers=1))
        self.layers = nn.ModuleList(layers)
        self.eps = eps

    def _get_args_kwargs(self):
        return self.args, self.kwargs

    def apply_constraints_(self):
        pass

    def forward(self, X):

        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        X_ = X.reshape((-1, n_freq, n_frames))

        # take absolute value, also makes complex value real
        # manual implementation of complex magnitude
        Y = bss.linalg.mag_sq(X_)

        Y = self.mel_layer(Y)

        for layer in self.layers:
            Y = layer(Y)

        # inverse mel filter bank (transpose)
        Y = self.fb[None, ...].matmul(Y)

        # apply sigmoid
        Y = pt.sigmoid(Y)

        # restore batch shape
        Y = Y.reshape(batch_shape + (n_freq, n_frames))

        # restore batch shape
        return Y
