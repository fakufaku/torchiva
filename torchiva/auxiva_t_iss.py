# Copyright (c) 2021 Robin Scheibler
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
"""
Joint Dereverberation and Blind Source Separation with Itarative Source Steering
================================================================================

Online implementation of the algorithm presented in [1]_.

References
----------
.. [1] T. Nakashima, R. Scheibler, M. Togami, and N. Ono,
    JOINT DEREVERBERATION AND SEPARATION WITH ITERATIVE SOURCE STEERING,
    ICASSP, 2021, https://arxiv.org/pdf/2102.06322.pdf.
"""
from typing import Optional, List, NoReturn, Tuple
import torch

from .linalg import hankel_view, mag_sq, divide, multiply
from .models import LaplaceModel
from .parameters import eps_models


def demix_derev(X, X_bar, W, H):
    reverb = torch.einsum("...cfdt,...dftn->...cfn", H, X_bar)
    sep = torch.einsum("...cfd,...dfn->...cfn", W, X)

    return sep - reverb


def iss_block_update_type_1(
    src: int, X: torch.Tensor, weights: torch.Tensor, eps: Optional[float] = 1e-3
) -> torch.Tensor:
    """
    Compute the update vector for ISS corresponding to update of the sources
    Equation (9) in [1]_
    """
    n_chan, n_freq, n_frames = X.shape[-3:]

    Xs = X[..., src, :, :]
    norm = 1.0 / n_frames

    v_num = torch.einsum("...cfn,...cfn,...fn->...cf", weights, X, Xs.conj()) * norm
    v_denom = torch.einsum("...cfn,...fn->...cf", weights, mag_sq(Xs)) * norm

    v = divide(v_num, v_denom, eps=eps)
    v_s = 1.0 - (1.0 / torch.sqrt(torch.clamp(v_denom[..., src, :], min=eps)))
    v[..., src, :] = v_s

    return v


def iss_block_update_type_2(
    src: int,
    tap: int,
    X: torch.Tensor,
    X_bar: torch.Tensor,
    weights: torch.Tensor,
    eps: Optional[float] = 1e-3,
) -> torch.Tensor:
    """
    Compute the update vector for ISS corresponding to update of the taps
    Equation (9) in [1]_
    """
    n_chan, n_freq, n_frames = X.shape[-3:]

    norm = 1.0 / n_frames

    Xst = X_bar[..., src, :, tap, :]

    v_num = torch.einsum("...cfn,...cfn,...fn->...cf", weights, X, Xst.conj()) * norm
    v_denom = torch.einsum("...cfn,...fn->...cf", weights, mag_sq(Xst)) * norm

    v = divide(v_num, v_denom, eps=eps)

    return v


def iss_updates(X, X_bar, W, weights, eps=1e-3):
    n_chan, n_freq, n_frames = X.shape[-3:]
    n_taps = X_bar.shape[-2]

    # source separation part
    for src in range(n_chan):
        v = iss_block_update_type_1(src, X, weights, eps=eps)
        X = X - torch.einsum("...cf,...fn->...cfn", v, X[..., src, :, :])
        W = W - torch.einsum("...cf,...fd->...cfd", v, W[..., src, :, :])

    # dereverberation part
    for src in range(n_chan):
        for tap in range(n_taps):
            v = iss_block_update_type_2(src, tap, X, X_bar, weights, eps=eps)
            X = X - torch.einsum("...cf,...fn->...cfn", v, X_bar[..., src, :, tap, :])

    return X, W


def iss_updates_with_H(
    X: torch.Tensor,
    X_bar: torch.Tensor,
    W: torch.Tensor,
    H: torch.Tensor,
    weights: torch.Tensor,
    eps: Optional[float] = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ISS updates performed in-place
    """
    n_chan, n_freq, n_frames = X.shape[-3:]
    n_taps = X_bar.shape[-2]

    # source separation part
    for src in range(n_chan):
        v = iss_block_update_type_1(src, X, weights, eps=eps)
        X = X - torch.einsum("...cf,...fn->...cfn", v, X[..., src, :, :])
        W = W - torch.einsum("...cf,...fd->...cfd", v, W[..., src, :, :])
        H = H - torch.einsum("...cf,...fdt->cfdt", v, H[..., src, :, :, :])

    # dereverberation part
    for src in range(n_chan):
        for tap in range(n_taps):
            v = iss_block_update_type_2(src, tap, X, X_bar, weights, eps=eps)
            X = X - torch.einsum("...cf,...fn->...cfn", v, X_bar[..., src, :, tap, :])
            HV = H[..., src, tap] - v
            H[..., src, tap] = HV

    return X, W, H


def projection_back(Y, W, ref_mic=0, eps=1e-6):
    # projection back (not efficient yet)
    batch_shape = Y.shape[:-3]
    n_chan, n_freq, n_frames = Y.shape[-3:]

    # projection back (not efficient yet)
    e1_shape = [1] * (len(batch_shape) + 1) + [n_chan, 1]
    e1 = W.new_zeros(e1_shape)
    e1[..., 0, 0] = 1.0
    eye = torch.eye(n_chan, n_chan)[None, ...].type_as(W)
    e1 = eye[..., :, [ref_mic]]
    WT = W.transpose(-3, -2)
    WT = WT.transpose(-2, -1)
    a = torch.linalg.solve(WT + eps * eye, e1)
    a = a.transpose(-3, -2)
    Y = Y * a

    return Y, a


class AuxIVA_T_ISS(torch.nn.Module):
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        n_taps: Optional[int] = 0,
        n_delay: Optional[int] = 0,
        n_iter: Optional[int] = 10,
        proj_back: Optional[bool] = True,
        eps: Optional[float] = None,
        checkpoints_iter: Optional[List[int]] = None,
    ):
        super().__init__()
        self.n_taps = n_taps
        self.n_delay = n_delay
        self.n_iter = n_iter
        self.proj_back = proj_back

        self.W = None
        self.W_inv = None
        self.X = None
        self.X_hankel = None

        if eps is None:
            self.eps = eps_models["laplace"]
        else:
            self.eps = eps

        if model is None:
            self.model = LaplaceModel()
        else:
            self.model = model
        assert callable(self.model)

        # metrology
        self.checkpoints_iter = checkpoints_iter
        self.checkpoints_list = []

    def forward(
        self,
        X: torch.Tensor,
        n_iter: Optional[int] = None,
        proj_back: Optional[bool] = None,
    ) -> torch.Tensor:
        """
        Parameters
        ----------
        X: torch.Tensor, (..., n_channels, n_frequencies, n_frames)
            The input signal
        n_iter: int, optional
            The number of iterations
        proj_back:
            Flag that indicates if we want to restore the scale
            of the signal by projection back

        Returns
        -------
        Y: torch.Tensor, (..., n_channels, n_frequencies, n_frames)
            The separated and dereverberated signal
        """
        batch_shape = X.shape[:-3]
        n_chan, n_freq, n_frames = X.shape[-3:]

        if n_iter is None:
            n_iter = self.n_iter

        if proj_back is None:
            proj_back = self.proj_back

        self.X = X.clone()

        # shape (..., n_chan, n_freq, n_taps + n_delay + 1, block_size)
        X_pad = torch.nn.functional.pad(X, (self.n_taps + self.n_delay, 0))
        self.X_hankel = hankel_view(X_pad, self.n_taps + self.n_delay + 1)
        X_bar = self.X_hankel[..., : -self.n_delay - 1, :]  # shape (c, f, t, b)

        # the demixing matrix
        self.W = self.X.new_zeros(batch_shape + (n_chan, n_freq, n_chan))
        eye = torch.eye(n_chan).type_as(self.W)
        self.W[...] = eye[:, None, :]

        for epoch in range(n_iter):

            if self.checkpoints_iter is not None and epoch in self.checkpoints_iter:
                self.checkpoints_list.append(self.X)

            # shape: (n_chan, n_freq, n_frames)
            # model takes as input a tensor of shape (..., n_frequencies, n_frames)
            weights = self.model(self.X)

            # we normalize the sources to have source to have unit variance prior to
            # computing the model
            g = torch.clamp(
                torch.mean(mag_sq(self.X), dim=(-2, -1), keepdim=True), min=self.eps
            )
            g_sqrt = torch.sqrt(g)
            self.X = divide(self.X, g_sqrt, eps=self.eps)
            self.W = divide(self.W, g_sqrt, eps=self.eps)
            weights = weights * g

            # Iterative Source Steering updates
            self.X, self.W = iss_updates(self.X, X_bar, self.W, weights, eps=self.eps)

        # projection back
        if proj_back:
            self.X, _ = projection_back(self.X, self.W, eps=self.eps)

        return self.X
