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

from typing import Optional, Union

import numpy as np
import torch
from scipy.signal import get_window, istft, stft


class STFT(torch.nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: Optional[int] = None,
        window: Optional[Union[str, np.ndarray, torch.Tensor]] = None,
        dtype=None,
    ):
        super().__init__()

        self._n_fft = n_fft
        self._n_freq = n_fft // 2 + 1
        self._hop_length = hop_length

        if dtype is None and (window is None or isinstance(window, str)):
            self.dtype = torch.zeros(1).dtype  # trick to get default type of torch
        else:
            self.dtype = dtype

        if window is None:
            window = "hann"

        if isinstance(window, str):
            self._window_type = window
            window = get_window(window, self._n_fft)
        else:
            self.window_type = "custom"

        if isinstance(window, np.ndarray):
            window = torch.from_numpy(window)

        # assign proper type, check dimension, and register as buffer
        window = window.to(self.dtype)
        assert isinstance(window, torch.Tensor)
        assert window.ndim == 1 and window.shape[0] == self._n_fft
        self.register_buffer("_window", window)

    @property
    def n_fft(self) -> int:
        return self._n_fft

    @property
    def hop_length(self) -> int:
        return self._hop_length

    @property
    def n_freq(self) -> int:
        return self._n_freq

    @property
    def window(self) -> torch.Tensor:
        return self._window

    @property
    def window_type(self) -> str:
        return self._window_type

    def forward(self, x):
        batch_shape = x.shape[:-1]
        n_samples = x.shape[-1]
        x = x.reshape((-1, n_samples))

        # transform! shape (n_batch * n_channels, n_frequencies, n_frames)
        # as of pytorch 1.8 the output can be complex
        x = torch.stft(
            x,
            self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )

        # restore batch shape
        x = x.reshape(batch_shape + x.shape[-2:])

        # make complex tensor and re-order shape (n_batch, n_channels, n_frequencies, n_frames)
        return x

    def inv(self, x):
        batch_shape = x.shape[:-2]
        n_freq, n_frames = x.shape[-2:]

        # flatten th batch and channels
        x = x.reshape((-1, n_freq, n_frames))

        # inverse transform on flat batch/channels
        x = torch.istft(
            x,
            self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=False,
        )

        # reshape as (n_batch, n_channels, n_samples) and return
        return x.reshape(batch_shape + x.shape[-1:])
