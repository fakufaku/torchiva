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

import math
from typing import List, Optional

import torch

from .base import SourceModelBase, DRBSSBase
from .linalg import bmm, divide, eigh, hermite, mag, mag_sq, multiply, inv_loaded
from .models import LaplaceModel
from .parameters import eps_models


def normalize(x):
    return x / torch.linalg.norm(x, dim=-2, keepdim=True)


def smallest_eigenvector_eigh_cpu(V):
    dev = V.device

    # diagonal loading factor
    # dload = 1e-5 * torch.diag(torch.arange(V.shape[-1], device=V.device))
    # V = (V + dload).cpu()
    V = V.cpu()
    r = torch.linalg.eigh(V)

    ev = r.eigenvalues[..., 0].to(dev)
    ev = torch.reciprocal(torch.clamp(ev, min=1e-5))

    x = r.eigenvectors[..., :, 0].to(dev)

    return ev, x


def smallest_eigenvector_power_method(V, n_iter=10):

    assert V.shape[-2] == V.shape[-1]

    # compute inverse of input matrix
    V_inv = inv_loaded(V, load=1e-5)

    # initial point x0 = [1, 0, ..., 0]
    x = V_inv[..., :, 0]
    x = normalize(x)

    for epoch in range(n_iter - 1):
        x = torch.einsum("...fcd,...fd->...fc", V_inv, x)
        ev = torch.linalg.norm(x, dim=-1)
        x = x / ev[..., None]

    # fix sign of first element to positive
    s = torch.conj(torch.sgn(x[..., 0]))
    x = x * s[..., None]

    return ev, x


def adjust_global_scale(Y, ref):
    num = torch.mean(torch.conj(Y) * ref, dim=(-2, -1), keepdim=True)
    denom = torch.mean(Y.abs().square(), dim=(-2, -1), keepdim=True)
    denom = torch.clamp(denom, min=1e-7)
    scale = num / denom
    return Y * scale


class FIVE(DRBSSBase):
    """
    Fast independent vector extraction (FIVE) [8]_.
    FIVE extracts one source from the input signal.

    Parameters
    ----------
    n_iter: int, optional
        The number of iterations (default: ``10``).
    model: torch.nn.Module, optional
        The model of source distribution (default: ``LaplaceModel``).
    proj_back_mic: int, optional
        The reference mic index to perform projection back.
        If set to ``None``, projection back is not applied (default: ``0``).
    eps: float, optional
        A small constant to make divisions and the like numerically stable (default: ``None``).
    n_power_iter: int, optional
        The number of power iterations.
        If set to ``None``, eigenvector decomposition is used instead. (default: ``None``)


    Methods
    --------
    forward(X, n_iter=None, model=None, proj_back_mic=None, eps=None)

    Parameters
    ----------
    X: torch.Tensor
        The input mixture in STFT-domain,
        ``shape (..., n_chan, n_freq, n_frames)``

    Returns
    -------
    Y: torch.Tensor, ``shape (..., n_freq, n_frames)``
        The extracted *one* signal in STFT-domain.


    References
    ---------
    .. [8] R. Scheibler, and N Ono,
        "Fast independent vector extraction by iterative SINR maximization",
        ICASSP, 2020, https://arxiv.org/pdf/1910.10654.pdf.

    """

    def __init__(
        self,
        n_iter: Optional[int] = 10,
        model: Optional[torch.nn.Module] = None,
        proj_back_mic: Optional[int] = 0,
        eps: Optional[float] = None,
        n_power_iter: Optional[int] = None,
    ):
        super().__init__(
            n_iter,
            model=model,
            proj_back_mic=proj_back_mic,
            eps=eps,
        )

        self.n_power_iter = n_power_iter

    def forward(
        self,
        X: torch.Tensor,
        n_iter: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
        proj_back_mic: Optional[int] = None,
        eps: Optional[float] = None,
        n_power_iter: Optional[int] = None,
    ) -> torch.Tensor:

        n_chan, n_freq, n_frames = X.shape[-3:]

        n_iter, model, proj_back_mic, eps, n_power_iter = self._set_params(
            n_iter=n_iter,
            model=model,
            proj_back_mic=proj_back_mic,
            eps=eps,
            n_power_iter=n_power_iter,
        )

        # for now, only supports determined case
        assert callable(model)

        # initialize source model if NMF
        self._reset(model)

        # remove DC part
        X = X[..., 1:, :]

        # Pre-whitening

        # covariance matrix of input signal (n_freq, n_chan, n_chan)
        Cx = torch.einsum("...cfn,...dfn->...fcd", X, X.conj()) / n_frames
        Cx = 0.5 * (Cx + hermite(Cx))

        # We will need the inverse square root of Cx
        # e_val, e_vec = torch.linalg.eigh(Cx)
        e_val, e_vec = eigh(Cx)

        # put the eigenvalues in descending order
        e_vec = torch.flip(e_vec, dims=[-1])
        e_val = torch.flip(e_val, dims=[-1])

        # compute the whitening matrix
        e_val_sqrt = torch.sqrt(torch.clamp(e_val, min=1e-5))
        Q = torch.reciprocal(e_val_sqrt[..., :, None]) * hermite(e_vec)
        Qinv = e_vec * e_val_sqrt[..., None, :]

        # we keep the row of Q^{-1} corresponding to the reference mic for the
        # normalization
        Qinv_ref = Qinv[..., proj_back_mic, :]
        # Qinv_ref = e_vec[..., proj_back_mic, None, :] * e_val_sqrt[..., None, :]

        # keep the input for scaling
        X_in = X

        # The whitened input signal
        X = torch.einsum("...fcd,...dfn->...cfn", Q, X)

        # Pick the initial signal as the largest component of PCA,
        # i.e., the first channel (because we flipped the order above)
        Y = X[..., 0, :, :]

        # Y = adjust_global_scale(Y, X_in[..., proj_back_mic, None, :, :])

        # projection back
        if proj_back_mic is not None:
            z = Q[..., 0, None, :] @ Cx[..., proj_back_mic, None]
            z = z.transpose(-3, -2)
            Y = z[..., 0, :, :] * Y

        for epoch in range(n_iter):

            # shape: (n_chan, n_freq, n_frames)
            # model takes as input a tensor of shape (..., n_src, n_masks, n_frequencies, n_frames)
            weights = model(Y)

            # compute the weighted spatial covariance matrix
            V = torch.einsum("...fn,...cfn,...dfn->...fcd", weights, X, X.conj())
            V = 0.5 * (V + hermite(V))

            # now compute the demixing vector
            # inv_eigenvalue (..., n_freq)
            # eigenvector (..., n_freq, n_chan)
            if n_power_iter is None:
                inv_eigenvalue, eigenvector = smallest_eigenvector_eigh_cpu(V)
            else:
                inv_eigenvalue, eigenvector = smallest_eigenvector_power_method(
                    V, n_iter=n_power_iter
                )

            # projection back
            z = torch.einsum("...fc,...fc->...f", Qinv_ref, eigenvector)
            w = eigenvector * torch.conj(z[..., None])

            # the new output
            Y = torch.einsum("...fc,...cfn->...fn", w.conj(), X)

        # add back DC offset
        pad_shape = Y.shape[:-2] + (1,) + Y.shape[-1:]
        Y = torch.cat((Y.new_zeros(pad_shape), Y), dim=-2)

        return Y[..., None, :, :]
