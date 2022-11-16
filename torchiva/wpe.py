# Copyright 2022 Robin Scheibler, Kohei Saijo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
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

from typing import Optional
import torch

from .linalg import hankel_view, mag_sq, solve_loaded
from .base import DRBSSBase


def derev(H: torch.Tensor, X: torch.Tensor, X_bar: torch.Tensor):
    return X - torch.einsum("...fcts,...cftn->...sfn", H.conj(), X_bar)


def wpe_default_weights(Y: torch.Tensor, eps: Optional[float] = 1e-5) -> torch.Tensor:
    w = 1.0 / torch.clamp(torch.mean(mag_sq(Y), dim=-3), min=eps)
    w = w / w.sum(dim=-1, keepdim=True)
    return w


def wpe_one_iter(
    Y: torch.Tensor,
    X: torch.Tensor,
    X_bar: torch.Tensor,
    model: Optional[callable] = None,
    eps: Optional[float] = 1e-5,
) -> torch.Tensor:
    """
    Parameters
    ----------
    Y: torch.Tensor, (..., n_chan, n_freq, n_frames)
        The current estimate of the dereverberated signal
    X: torch.Tensor, (..., n_chan, n_freq, n_frames)
        Input signal
    X_bar: torch.Tensor, (..., n_chan, n_freq, n_taps, n_frames)
        Delayed version of input signal

    Returns
    -------
    H: torch.Tensor, (..., n_freq, n_chan, n_taps, n_chan)
        The updated dereverberation filter weights
    """
    batch_shape = X_bar.shape[:-4]
    n_chan, n_freq, n_taps, n_frames = X_bar.shape[-4:]
    Lh = n_taps * n_chan

    if model is None:
        model = wpe_default_weights
        weights = model(Y, eps=eps)
    else:
        weights = model(Y)

    # compute weighted statistics
    acm = torch.einsum("...fn,...cftn,...dfun->...fctdu", weights, X_bar, X_bar.conj())
    xcv = torch.einsum("...fn,...cftn,...sfn->...fcts", weights, X_bar, X.conj())

    # solve the system
    acm = acm.reshape(batch_shape + (n_freq, Lh, Lh))
    xcv = xcv.reshape(batch_shape + (n_freq, Lh, n_chan))

    # H = torch.linalg.solve(acm + eps * torch.eye(Lh).type_as(acm), xcv)
    # H = torch.linalg.solve(acm, xcv)
    H = solve_loaded(acm, xcv, load=eps)

    H = H.reshape(batch_shape + (n_freq, n_chan, n_taps, n_chan))

    return H


class WPE(DRBSSBase):
    """
    Weighted prediction error (WPE) [9]_.

    Parameters
    ----------
    n_iter: int, optional
        The number of iterations. (default: ``3``)
    n_taps: int, optional
        The length of the dereverberation filter (default: ``5``).
    n_delay: int, optional
        The number of delay for dereverberation (default: ``3``).
    model: torch.nn.Module, optional
        The model of source distribution.
        If ``None``, time-varying Gaussian is used. (default: ``None``).
    eps: float, optional
        A small constant to make divisions and the like numerically stable (default:``1e-5``).

    Returns
    ----------
    Y: torch.Tensor, ``shape (..., n_src, n_freq, n_frames)``
        The dereverberated signal in STFT-domain.


    References
    ---------
    .. [9] T. Nakatani, T. Yoshioka, K. Kinoshita, M. Miyoshi, and B. H. Juang,
        "Speech dereverberation based on variance-normalized delayed linear prediction",
        IEEE Trans. on Audio, Speech, and Lang. Process., 2010.
    """

    def __init__(
        self,
        n_iter: Optional[int] = 3,
        n_delay: Optional[int] = 3,
        n_taps: Optional[int] = 5,
        model: Optional[torch.nn.Module] = None,
        eps: Optional[float] = 1e-5,
    ):

        super().__init__(
            n_iter,
            n_taps=n_taps,
            n_delay=n_delay,
            eps=eps,
        )

        self.model = model

    def forward(
        self,
        X: torch.Tensor,
        n_iter: Optional[int] = None,
        n_delay: Optional[int] = None,
        n_taps: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
        eps: Optional[float] = None,
    ):

        n_iter, n_taps, n_delay, model, eps = self._set_params(
            n_iter=n_iter,
            n_taps=n_taps,
            n_delay=n_delay,
            model=model,
            eps=eps,
        )

        batch_shape = X.shape[:-3]
        n_chan, n_freq, n_frames = X.shape[-3:]

        # shape (..., n_chan, n_freq, n_taps + n_delay + 1, block_size)
        X_pad = torch.nn.functional.pad(X, (n_taps + n_delay, 0))
        X_hankel = hankel_view(X_pad, n_taps + n_delay + 1)
        X_bar = X_hankel[..., : -n_delay - 1, :]  # shape (c, f, t, b)

        Y = X.clone()

        for epoch in range(n_iter):
            H = wpe_one_iter(Y, X, X_bar, model=model, eps=eps)
            Y = derev(H, X, X_bar)

        return Y
