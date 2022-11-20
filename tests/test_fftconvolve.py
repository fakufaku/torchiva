import pytest
import torch
import torchiva
from scipy.signal import fftconvolve

x = torch.zeros((100)).normal_()
t = torch.zeros((101)).normal_()
y = torch.zeros((21)).normal_()
z = torch.zeros((20)).normal_()

@pytest.mark.parametrize(
    "x1, x2, mode, dim",
    [
        (x, y, "full", -1),
        (x, y, "same", -1),
        (x, y, "valid", -1),
        (y, x, "full", -1),
        (y, x, "same", -1),
        (y, x, "valid", -1),
        (x, z, "full", -1),
        (x, z, "same", -1),
        (x, z, "valid", -1),
        (z, x, "full", -1),
        (z, x, "same", -1),
        (z, x, "valid", -1),
        (t, y, "full", -1),
        (t, y, "same", -1),
        (t, y, "valid", -1),
        (y, t, "full", -1),
        (y, t, "same", -1),
        (y, t, "valid", -1),
        (t, z, "full", -1),
        (t, z, "same", -1),
        (t, z, "valid", -1),
        (z, t, "full", -1),
        (z, t, "same", -1),
        (z, t, "valid", -1),
    ],
)
def test_fftconvolve(x1, x2, mode, dim):
    y_t = torchiva.fftconvolve(x1, x2, mode).numpy()
    y_n = fftconvolve(x1.numpy(), x2.numpy(), mode)

    assert abs(y_t - y_n).max() < 1e-5
