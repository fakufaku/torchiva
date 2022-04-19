# Copyright (c) 2022 Robin Scheibler
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
