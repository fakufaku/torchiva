from typing import Optional, Union

import torch as pt
from scipy.signal import get_window, istft, stft

from .base import STFTBase, Window
from .dtypes import dtype_cpx2f, dtype_f2cpx

_torch_windows = {
    Window.BARTLETT: pt.bartlett_window,
    Window.BLACKMAN: pt.blackman_window,
    Window.HAMMING: pt.hamming_window,
    Window.HANN: pt.hann_window,
}


class STFT(STFTBase):
    def __init__(
        self,
        n_fft: int,
        hop_length: Optional[int] = None,
        window: Optional[Union[Window, pt.Tensor]] = None,
    ):

        if isinstance(window, pt.Tensor):
            self._window_custom = window
            window = Window.CUSTOM
        else:
            self._window_custom = None

        super().__init__(n_fft, hop_length=hop_length, window=window)

    def _make_window(self, x):

        x_dtype = dtype_cpx2f(x)
        x_device = x.device

        if (
            self._window is not None
            and self._window.dtype == x_dtype
            and self._window.device == x_device
        ):
            return

        if self._window_type == Window.CUSTOM:
            _window = self._window_custom
            assert _window is not None, (
                "For custom windows, the numpy array of the window"
                " must be passed directly"
            )
            assert _window.ndim == 1, "The window must be a 1D array"
            assert (
                _window.shape[0] == self.n_fft
            ), "The window length must be equal to the FFT length"
            self._window = pt.Tensor(_window, dtype=x_dtype, device=x_device)
        else:
            self._window = _torch_windows[self._window_type](
                self.n_fft, dtype=x_dtype, device=x_device
            )

    def _forward(self, x: pt.Tensor):
        batch_shape = x.shape[:-1]
        n_samples = x.shape[-1]
        x_flat = x.reshape((-1, n_samples))

        # transform! shape (n_batch * n_channels, n_frequencies, n_frames)
        # as of pytorch 1.8 the output can be complex
        X_flat = pt.stft(
            x_flat,
            self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=True,
        )

        # restore batch shape
        X = X_flat.reshape(batch_shape + X_flat.shape[-2:])

        # make complex tensor and re-order shape (n_batch, n_channels, n_frequencies, n_frames)
        return X

    def _backward(self, X: pt.Tensor):
        batch_shape = X.shape[:-2]
        n_freq, n_frames = X.shape[-2:]

        # flatten th batch and channels
        X_flat = X.reshape((-1, n_freq, n_frames))

        # inverse transform on flat batch/channels
        x_flat = pt.istft(
            X_flat,
            self.n_fft,
            hop_length=self.hop_length,
            window=self.window,
            return_complex=False,
        )

        # reshape as (n_batch, n_channels, n_samples) and return
        return x_flat.reshape(batch_shape + x_flat.shape[-1:])
