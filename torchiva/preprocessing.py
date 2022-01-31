from typing import Optional, Union

import torch

from .base import Window
from .stft import STFT


def filter_dc(
    x: torch.Tensor,
    n_fft: Optional[int] = 1024,
    hop_length: Optional[int] = 256,
    window: Optional[Union[str, Window]] = Window.HAMMING,
    fc_cut: Optional[float] = 300,
    fs: Optional[float] = 16000,
):
    """
    STFT domain filtering of the DC component

    Parameters
    ----------
    x: torch.Tensor (..., n_samples)
        The input signal to filter
    n_fft: int, optional
        The FFT length to use for the STFT
    hop_length: int, optional
        The shift to use for the STFT
    window: torchiva.Window or str, optional
        The window to use for the STFT

    Returns
    -------
    torch.Tensor (..., n_samples)
    """
    n_samples = x.shape[-1]

    # compute the filter
    n_bins = int(fc_cut / fs * n_fft)
    coeffs = torch.hamming_window(n_bins * 2)[:n_bins].type_as(x)
    coeffs = coeffs[:, None]

    stft = STFT(n_fft=1024, hop_length=256, window=Window.HAMMING)
    X = stft(x)
    X[..., :n_bins, :] *= coeffs
    x = stft.inv(X)

    return x[..., :n_samples]
