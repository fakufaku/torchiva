import torch
import math
from torch import nn
import torch.nn.functional as F


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


class ComplexLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape,
        eps=1e-05,
        elementwise_affine=True,
        device=None,
        dtype=None,
    ):
        super().__init__()

        if isinstance(normalized_shape, int):
            self.normalized_shape = [normalized_shape]
        else:
            self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine
        self.eps = eps

        self.norm_dim = list(range(-len(normalized_shape), 0))

        if dtype is None:
            dtype = torch.complex64

        if self.elementwise_affine:
            self.gamma = torch.nn.Parameter(
                torch.zeros(self.normalized_shape, dtype=dtype, device=device)
            )
            self.beta = torch.nn.Parameter(
                torch.zeros(self.normalized_shape, dtype=dtype, device=device)
            )

    def forward(self, x):

        var = torch.var(x, dim=self.norm_dim, keepdim=True)
        mean = torch.mean(x, dim=self.norm_dim, keepdim=True)

        x = (x - mean) / torch.sqrt(var + self.eps)

        if self.elementwise_affine:
            x = self.gamma * x + self.beta

        return x


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


def complex_non_linearity2(x):
    mod = x.abs()
    G = torch.sigmoid(5 * (mod - 1.0)) / torch.clamp(mod, min=1e-5)
    return G * x


def complex_gelu(x):
    if x.dtype in [torch.complex64, torch.complex128]:
        g = F.gelu(x.real) / torch.clamp(x.real, min=1e-6)
        return g * x
    else:
        return F.gelu(x)


class FeedForward(nn.Module):
    def __init__(
        self, num_features, expansion_factor, dropout, complex=False, use_relu=False
    ):
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
        if self.complex:
            x = complex_non_linearity2(x)
        else:
            x = F.gelu(x)
        x = self.dropout1(x)
        out = self.dropout2(self.fc2(x))
        return out


class ComplexFNetEncoderLayer(nn.Module):
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


class ComplexFNet(nn.TransformerEncoder):
    def __init__(
        self,
        d_model=256,
        expansion_factor=2,
        dropout=0.5,
        num_layers=6,
    ):
        encoder_layer = ComplexFNetEncoderLayer(d_model, expansion_factor, dropout)
        super().__init__(encoder_layer=encoder_layer, num_layers=num_layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x
