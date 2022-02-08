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
from typing import List, NoReturn, Optional, Tuple

import torch
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .linalg import divide, hankel_view, hermite, mag_sq, multiply
from .models import LaplaceModel
from .parameters import eps_models


def demix_derev(X, X_bar, W, H):
    reverb = torch.einsum("...cfdt,...dftn->...cfn", H, X_bar)
    sep = torch.einsum("...cfd,...dfn->...cfn", W, X)
    return sep - reverb


def demix_background(X, J):
    n_src = J.shape[-1]
    return (
        torch.einsum("...cfd, ...dfn->...cfn", J, X[..., :n_src, :, :])
        - X[..., n_src:, :, :]
    )


def iss_block_update_type_1(
    src: int,
    X: torch.Tensor,
    weights: torch.Tensor,
    n_src: Optional[int] = None,
    eps: Optional[float] = 1e-3,
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
    X: torch.Tensor,
    Zs: torch.Tensor,
    weights: torch.Tensor,
    eps: Optional[float] = 1e-3,
) -> torch.Tensor:
    """
    Compute the update vector for ISS corresponding to update of the taps
    Equation (9) in [1]_
    """
    n_chan, n_freq, n_frames = X.shape[-3:]

    v_num = torch.einsum("...cfn,...cfn,...fn->...cf", weights, X, Zs.conj())
    v_denom = torch.einsum("...cfn,...fn->...cf", weights, mag_sq(Zs))

    v = divide(v_num, v_denom, eps=eps)

    return v


def iss_block_update_type_3(
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

    Xst = X_bar[..., src, :, tap, :]

    v_num = torch.einsum("...cfn,...cfn,...fn->...cf", weights, X, Xst.conj())
    v_denom = torch.einsum("...cfn,...fn->...cf", weights, mag_sq(Xst))

    v = divide(v_num, v_denom, eps=eps)

    return v


def iss_updates_with_H(
    X: torch.Tensor,
    X_bar: torch.Tensor,
    W: torch.Tensor,
    H: torch.Tensor,
    weights: torch.Tensor,
    J: Optional[torch.Tensor] = None,
    Z: Optional[torch.Tensor] = None,
    eps: Optional[float] = 1e-3,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    ISS updates performed in-place

    Parameters
    ----------
    X: torch.Tensor (..., n_src, n_freq, n_frames)
        Separated signals
    Z: torch.Tensor (..., n_chan - n_src, n_freq, n_frames)
    X_bar: torch.Tensor (..., )
        Delayed versions of the input signal
    W: torch.Tensor (..., n_src, n_freq, n_chan)
        The demixing matrix part corresponding to target sources
    H: torch.Tensor (..., n_src, n_freq, n_chan, n_taps)
        The dereverberation matrix
    J: torch.Tensor (..., n_chan - n_src, n_freq, n_src)
        The demixing matrix part corresponding to background
    weights: torch.Tensor (..., n_src, n_freq, n_frames)
        The separation masks
    n_src: int, optional
        The number of target sources
    eps: float, optional
        A small constant used for numerical stability
    """
    n_src, n_freq, n_frames = X.shape[-3:]
    if J is None:
        n_chan = n_src
    else:
        n_chan = n_src + Z.shape[-3]
    n_taps = X_bar.shape[-2]

    # we make a copy because we need to do some inplace operations
    H = H.clone()
    W = W.clone()

    # source separation part
    for src in range(n_src):
        v = iss_block_update_type_1(src, X, weights, eps=eps)
        X = X - torch.einsum("...cf,...fn->...cfn", v, X[..., src, :, :])
        W = W - torch.einsum("...cf,...fd->...cfd", v, W[..., src, :, :])
        H = H - torch.einsum("...cf,...fdt->...cfdt", v, H[..., src, :, :, :])

    # background part
    if Z is not None and J is not None:
        for bak in range(n_chan - n_src):
            v = iss_block_update_type_2(X, Z[..., bak, :, :], weights, eps=eps)
            X = X - torch.einsum("...cf,...fn->...cfn", v, Z[..., bak, :, :])
            W[..., :n_src] = W[..., :n_src] - torch.einsum(
                "...cf,...fd->...cfd", v, J[..., bak, :, :]
            )
            W[..., n_src + bak] = W[..., n_src + bak] + v

    # dereverberation part
    for src in range(n_chan):
        for tap in range(n_taps):
            v = iss_block_update_type_3(src, tap, X, X_bar, weights, eps=eps)
            X = X - torch.einsum("...cf,...fn->...cfn", v, X_bar[..., src, :, tap, :])
            HV = H[..., src, tap] + v
            H[..., src, tap] = HV

    return X, W, H


def background_update(W, H, J, X, C_XX, C_XbarX):
    """
    Recomputes J based on W, H, and C_XX = E[X X^H] and C_XbarX = E[ X_bar X^H ]
    """
    n_src, n_freq, n_chan = W.shape[-3:]

    A1 = torch.einsum("...sfc,...cfd->...fsd", W, C_XX)
    A2 = torch.einsum("...sfdt,...dtfc->...fsc", H, C_XbarX)
    A = A1 + A2  # (..., n_freq, n_src, n_chan)
    J_H = torch.linalg.solve(A[..., :n_src], A[..., n_src:])
    # J = J_H.conj().permute([-1, -3, -2])
    J = J_H.conj().moveaxis(-1, -3)  # (..., n_chan - n_src, n_freq, n_src)

    return J


def over_iss_t_one_iter(Y, X, X_bar, C_XX, C_XbarX, W, H, J, model, eps=1e-3):
    # shape: (n_chan, n_freq, n_frames)
    # model takes as input a tensor of shape (..., n_frequencies, n_frames)
    weights = model(Y)

    # we normalize the sources to have source to have unit variance prior to
    # computing the model
    g = torch.clamp(torch.mean(mag_sq(Y), dim=(-2, -1), keepdim=True), min=eps)
    g_sqrt = torch.sqrt(g)
    Y = divide(Y, g_sqrt, eps=eps)
    W = divide(W, g_sqrt, eps=eps)
    H = divide(H, g_sqrt[..., None], eps=eps)
    weights = weights * g

    # Update the background part
    if J is not None:
        J = background_update(W, H, J, X, C_XX, C_XbarX)
        Z = demix_background(X, J)  # Z is None if J is None
    else:
        Z = None

    # Iterative Source Steering updates
    Y, W, H = iss_updates_with_H(Y, X_bar, W, H, weights, Z=Z, J=J, eps=eps)

    return Y, W, H, J


def over_iss_t_one_iter_dmc(X, X_bar, C_XX, C_XbarX, W, H, J, model, eps=1e-3):

    Y = demix_derev(X, X_bar, W, H)

    Y, W, H, J = over_iss_t_one_iter(
        Y, X, X_bar, C_XX, C_XbarX, W, H, J, model, eps=eps
    )

    return W, H, J


def projection_back_weights(W, J=None, ref_mic=0, eps=1e-6):
    # projection back (not efficient yet)
    n_src, n_freq, n_chan = W.shape[-3:]

    eye = torch.eye(n_src).type_as(W)

    if n_src < n_chan:
        # overdetermined case
        assert J is not None

        W1, W2 = W[..., :, :, :n_src], W[..., :, :, n_src:]
        B = torch.einsum("...sfc,...cfk->...sfk", W2, J)
        B = B + W1

        if ref_mic < n_src:
            rhs = eye[:, [ref_mic]]
        else:
            rhs = J[..., [ref_mic - n_src], :, :]
            rhs = rhs.permute([-2, -1, -3])

        BT = torch.einsum("...sfc->...fcs", B)
        a = torch.linalg.solve(BT + eps * eye, rhs)
        a = a.transpose(-3, -2)

    else:
        # determined case
        e1 = eye[..., :, [ref_mic]]
        WT = W.transpose(-3, -2)
        WT = WT.transpose(-2, -1)
        a = torch.linalg.solve(WT + eps * eye, e1)
        a = a.transpose(-3, -2)

    return a


class OverISS_T(torch.nn.Module):
    def __init__(
        self,
        model: Optional[torch.nn.Module] = None,
        n_taps: Optional[int] = 5,
        n_delay: Optional[int] = 1,
        n_iter: Optional[int] = 20,
        proj_back: Optional[bool] = True,
        ref_mic: Optional[bool] = 0,
        use_dmc: Optional[bool] = False,
        eps: Optional[float] = None,
    ):
        super().__init__()

        self.n_taps = n_taps
        self.n_delay = n_delay
        self.n_iter = n_iter
        self.proj_back = proj_back
        self.ref_mic = ref_mic
        self.use_dmc = use_dmc

        # the different parts of the demixing matrix
        self.W = None  # target sources
        self.H = None  # reverb for target sources
        self.J = None  # background in overdetermined case

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
        self.checkpoints_list = []

    def forward(
        self,
        X: torch.Tensor,
        n_src: Optional[int] = None,
        n_iter: Optional[int] = None,
        n_taps: Optional[int] = None,
        n_delay: Optional[int] = None,
        proj_back: Optional[bool] = None,
        use_dmc: Optional[bool] = None,
        checkpoints_iter: Optional[List] = None,
        checkpoints_list: Optional[List] = None,
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

        if n_src is None:
            n_src = n_chan
        elif n_src > n_chan:
            raise ValueError(
                f"Underdetermined source separation (n_src={n_src}, n_channels={n_chan})"
                f" is not supported"
            )

        is_overdet = n_src < n_chan

        if n_iter is None:
            n_iter = self.n_iter

        if proj_back is None:
            proj_back = self.proj_back

        if use_dmc is None:
            use_dmc = self.use_dmc

        self.checkpoints_iter = checkpoints_iter

        # shape (..., n_chan, n_freq, n_taps + n_delay + 1, block_size)
        X_pad = torch.nn.functional.pad(X, (self.n_taps + self.n_delay, 0))
        X_hankel = hankel_view(X_pad, self.n_taps + self.n_delay + 1)
        X_bar = X_hankel[..., : -self.n_delay - 1, :]  # shape (c, f, t, b)

        if is_overdet:
            C_XX = torch.einsum("...cfn,...dfn->...cfd", X, X.conj()) / X.shape[-1]
            C_XbarX = (
                torch.einsum("...cftn,...dfn->...ctfd", X_bar, X.conj()) / X.shape[-1]
            )
        else:
            C_XX, C_XbarX = None, None

        # the demixing matrix
        W = X.new_zeros(batch_shape + (n_src, n_freq, n_chan))
        if is_overdet:
            J = X.new_zeros(batch_shape + (n_chan - n_src, n_freq, n_src))
        else:
            J = None
        eye = torch.clamp(torch.eye(n_src, n_chan), min=self.eps).type_as(W)
        W[...] = eye[:, None, :]

        H = X.new_zeros(batch_shape + (n_src, n_freq, n_chan, self.n_taps))

        Y = demix_derev(X, X_bar, W, H)

        for epoch in range(n_iter):

            if self.checkpoints_iter is not None and epoch in self.checkpoints_iter:
                # self.checkpoints_list.append(X)
                if epoch == 0:
                    checkpoints_list.append(Y.clone().detach())
                else:
                    a = projection_back_weights(
                        W, J=J, eps=self.eps, ref_mic=self.ref_mic
                    )
                    checkpoints_list.append((Y * a).detach())

            if use_dmc:
                W, H, J = torch_checkpoint(
                    over_iss_t_one_iter_dmc,
                    X,
                    X_bar,
                    C_XX,
                    C_XbarX,
                    W,
                    H,
                    J,
                    self.model,
                    self.eps,
                    preserve_rng_state=True,
                )
            else:
                Y, W, H, J = over_iss_t_one_iter(
                    Y,
                    X,
                    X_bar,
                    C_XX,
                    C_XbarX,
                    W,
                    H,
                    J,
                    self.model,
                    eps=self.eps,
                )
                # Y = demix_derev(X, X_bar, W, H)

        # projection back
        if proj_back:
            a = projection_back_weights(W, J=J, eps=self.eps, ref_mic=self.ref_mic)
            if use_dmc:
                Y = a * demix_derev(X, X_bar, W, H)
            else:
                Y = a * Y

        return Y
