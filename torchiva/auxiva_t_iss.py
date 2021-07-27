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
from typing import Optional, List
import torch

from .linalg import hankel_view, mag_sq, divide, multiply
from .models import LaplaceModel
from .parameters import eps_models


def iss_block_update_type_1(
    src: int, X: torch.Tensor, weights: torch.Tensor
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

    v = divide(v_num, v_denom, eps=1e-3)
    v_s = 1.0 - (1.0 / torch.sqrt(torch.clamp(v_denom[..., src, :], min=1e-3)))
    v[..., src, :] = v_s

    return v


def iss_block_update_type_2(
    src: int, tap: int, X: torch.Tensor, X_bar: torch.Tensor, weights: torch.Tensor
) -> torch.Tensor:
    """
    Compute the update vector for ISS corresponding to update of the taps
    Equation (9) in [1]_
    """
    n_chan, n_freq, n_frames = X.shape[-3:]

    norm = 1.0 / n_frames

    Xst = X_bar[..., src, :, tap, :]

    v_num = torch.einsum("...cfn,...cfn,...fn->...cf", weights, X, Xst.conj()) * norm
    v_denom = torch.einsum("...cfn,...fn->...cf", weights, mag_sq(Xst))

    v = divide(v_num, v_denom, eps=1e-3)

    return v


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

        if model is None:
            self.model = LaplaceModel()
        else:
            self.model = model

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

        # the dereverberation filters
        self.h = self.X.new_zeros(batch_shape + (n_chan, n_freq, n_chan, self.n_taps))

        for epoch in range(n_iter):

            if self.checkpoints_iter is not None and epoch in self.checkpoints_iter:
                self.checkpoints_list.append(X)

            # shape: (n_chan, n_freq, n_frames)
            # model takes as input a tensor of shape (..., n_frequencies, n_frames)
            weights = self.model(X)

            # we normalize the sources to have source to have unit variance prior to
            # computing the model
            g = torch.clamp(torch.mean(mag_sq(X), dim=(-2, -1), keepdim=True), min=1e-5)
            X = divide(X, torch.sqrt(g))
            weights = weights * g

            # Iterative Source Steering updates

            # source separation part
            for src in range(n_chan):
                v = iss_block_update_type_1(src, self.X, weights)
                self.X = self.X - torch.einsum(
                    "...cf,...fn->...cfn", v, self.X[..., src, :, :]
                )
                self.W = self.W - torch.einsum(
                    "...cf,...fd->...cfd", v, self.W[..., src, :, :]
                )
                # self.h = self.h - torch.einsum("...cf,...fdt->cfdt", v, self.h[..., src, :, :, :])

            # dereverberation part
            for src in range(n_chan):
                for tap in range(self.n_taps):
                    v = iss_block_update_type_2(src, tap, self.X, X_bar, weights)
                    self.X = self.X - torch.einsum(
                        "...cf,...fn->...cfn", v, X_bar[..., src, :, tap, :]
                    )
                    # hv = self.h[..., src, tap] - v[..., None, None]
                    # self.h[..., src, tap] = hv

        # projection back
        if proj_back:
            # projection back (not efficient yet)
            e1_shape = [1] * (len(batch_shape) + 1) + [n_chan, 1]
            e1 = self.W.new_zeros(e1_shape)
            e1[..., 0, 0] = 1.0
            e1 = torch.eye(n_chan, 1, dtype=self.W.dtype, device=self.W.device)[
                None, ...
            ]
            WT = self.W.transpose(-3, -2)
            WT = WT.transpose(-2, -1)
            a = torch.linalg.solve(WT, e1)
            a = a.transpose(-3, -2)
            self.X = self.X * a

        return self.X
