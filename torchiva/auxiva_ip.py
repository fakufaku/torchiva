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

from typing import List, Optional
import torch

from .linalg import (
    bmm,
    divide,
    eigh,
    hermite,
    mag,
    mag_sq,
    multiply,
    solve_loaded,
    solve_loaded_general,
)
from .base import DRBSSBase


def orthogonal_constraint(W, Cx, load=1e-4):
    n_src, n_chan = W.shape[-2:]

    if n_src == n_chan:
        return None
    elif n_chan < n_src:
        raise ValueError(
            "OverIVA requires the number of sources to be "
            "less or equal than that of microphones"
        )

    A = W @ Cx

    J_H = solve_loaded_general(A[..., :n_src], A[..., n_src:], load=load, eps=load)

    return J_H.transpose(-2, -1).conj()


def projection_back_from_demixing_matrix(
    Y: torch.Tensor,
    W: torch.Tensor,
    J: Optional[torch.Tensor] = None,
    ref_mic: Optional[int] = 0,
    load: Optional[float] = 1e-4,
) -> torch.Tensor:
    """
    Parameters
    ----------
    Y: torch.Tensor (..., n_channels, n_frequencies, n_frames)
        The demixed signals
    W: torch.Tensor (..., n_frequencies, n_channels, n_channels)
        The demixing matrix
    ref_mic: int, optional
        The reference channel
    eps: float, optional
        A diagonal loading factor for the solve method
    """
    # projection back (not efficient yet)
    batch_shape = Y.shape[:-3]
    n_src, n_freq, n_frames = Y.shape[-3:]
    n_chan = W.shape[-1]

    if J is None:
        eye = torch.eye(n_chan).type_as(W)
        e1 = eye[:, [ref_mic]]
        e1 = torch.broadcast_to(e1, W.shape[:-2] + e1.shape)
        a = solve_loaded_general(W.transpose(-2, -1), e1)  # (..., n_freq, n_chan, 1)
        a = a.transpose(-3, -2)

    else:
        A = W[..., :n_src, :n_src]
        B = W[..., :n_src, n_src:]
        C = J

        if ref_mic < n_src:
            eye = torch.eye(n_src, n_src).type_as(W)
            e1 = eye[:, [ref_mic]]
        else:
            e1 = C[..., [ref_mic - n_src], :].transpose(-2, -1)

        WW = A + B @ C
        e1 = torch.broadcast_to(e1, WW.shape[:-2] + e1.shape)
        a = solve_loaded_general(WW.transpose(-2, -1), e1)
        a = a.transpose(-3, -2)

    Y = Y * a

    return Y, a


def cost(model, Y, W, J=None, g=None):
    n_freq, n_src, n_chan = W.shape
    n_frames = Y.shape[-1]

    if J is not None:
        eye = torch.eye(n_chan - n_src).type_as(W)
        eye = torch.broadcast_to(eye, W.shape[:-2] + eye.shape)
        W = torch.cat((W, torch.cat((J, -eye), dim=-1)), dim=-2)

    c1 = torch.linalg.norm(Y, dim=-2).sum(dim=(-1, -2))
    _, c2 = torch.linalg.slogdet(W)

    cost = c1 - 2 * n_frames * c2.sum(dim=-1)

    return cost


class AuxIVA_IP(DRBSSBase):
    """
    Independent vector analysis (IVA) with iterative projection (IP) update [5]_.

    We do not support ILRMA-T with IP updates.


    Parameters
    ----------
    n_iter: int, optional
        The number of iterations. (default: ``10``)
    n_src: int, optional
        The number of sources to be separated.
        When ``n_src < n_chan``, a computationally cheaper variant (OverIVA) [6]_ is used.
        If set to ``None``,  ``n_src`` is set to ``n_chan`` (default: ``None``)
    model: torch.nn.Module, optional
        The model of source distribution.
        If ``None``, spherical Laplace is used (default: ``None``).
    proj_back_mic: int, optional
        The reference mic index to perform projection back.
        If set to ``None``, projection back is not applied (default: ``0``).
    eps: float, optional
        A small constant to make divisions and the like numerically stable (default:``None``).


    Methods
    --------
    forward(X, n_iter=None, n_src=None, model=None, proj_back_mic=None, eps=None)

    Parameters
    ----------
    X: torch.Tensor
        The input mixture in STFT-domain,
        ``shape (..., n_chan, n_freq, n_frames)``

    Returns
    ----------
    Y: torch.Tensor, ``shape (..., n_src, n_freq, n_frames)``
        The separated signal in STFT-domain


    Note
    ----
    This class can handle two BSS methods with IP update rule depending on the specified arguments:
        * AuxIVA-IP: ``n_chan==n_src, model=LaplaceMoldel() or GaussMoldel()``
        * ILRMA-IP: ``n_chan==n_src, model=NMFModel()``
        * OverIVA_IP [6]_: ``n_taps=0, n_delay=0, n_chan==n_src, model=NMFModel()``


    References
    ---------
    .. [5] N. Ono,
        "Stable and fast update rules for independent vector analysis based on auxiliary function technique",
        WASSPA, 2011.

    .. [6] R. Scheibler, and N Ono,
        "Independent vector analysis with more microphones than sources",
        WASSPA, 2019, https://arxiv.org/pdf/1905.07880.pdf.
    """

    def __init__(
        self,
        n_iter: Optional[int] = 10,
        n_src: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
        proj_back_mic: Optional[int] = 0,
        eps: Optional[float] = None,
    ):

        super().__init__(
            n_iter,
            n_src=n_src,
            model=model,
            proj_back_mic=proj_back_mic,
            eps=eps,
        )

    def forward(
        self,
        X: torch.Tensor,
        n_iter: Optional[int] = None,
        n_src: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
        proj_back_mic: Optional[int] = None,
        eps: Optional[float] = None,
        verbose: Optional[bool] = False,
    ) -> torch.Tensor:

        n_iter, n_src, model, proj_back_mic, eps = self._set_params(
            n_iter=n_iter,
            n_src=n_src,
            model=model,
            proj_back_mic=proj_back_mic,
            eps=eps,
        )

        batch_shape = X.shape[:-3]
        n_chan, n_freq, n_frames = X.shape[-3:]

        # for now, only supports determined case
        assert callable(model)

        # initialize source model if NMF
        self._reset(model)

        W = X.new_zeros(batch_shape + (n_freq, n_src, n_chan))
        W[:] = torch.eye(n_src, n_chan)
        J = None

        # initial estimate
        Y = X[..., :n_src, :, :].clone()  # sign to be consistant with W

        # covariance matrix of input signal (n_freq, n_chan, n_chan)
        Cx = torch.einsum("...cfn,...dfn->...fcd", X, X.conj()) / n_frames

        J = orthogonal_constraint(W, Cx, load=eps)

        if verbose:
            print(cost(model, Y, W, J))

        evec = torch.eye(n_chan).type_as(W)
        evec = torch.broadcast_to(evec, W.shape[:-2] + evec.shape)

        for epoch in range(n_iter):

            # shape: (n_chan, n_freq, n_frames)
            # model takes as input a tensor of shape (..., n_frequencies, n_frames)
            weights = model(Y)

            # we normalize the sources to have source to have unit variance prior to
            # computing the model
            g = torch.clamp(torch.mean(mag_sq(Y), dim=(-2, -1), keepdim=True), min=eps)
            g_sqrt = torch.sqrt(g)
            Y = Y / torch.clamp(g_sqrt, min=eps)
            W = W / torch.clamp(g_sqrt.transpose(-3, -2), min=eps)
            weights = weights * g

            for k in range(n_src):

                # apply the orthogonal constraint
                J = orthogonal_constraint(W, Cx, load=eps)

                # compute the weighted spatial covariance matrix
                V = torch.einsum(
                    "...cfn,...dfn->...fcd", X * weights[..., [k], :, :], X.conj()
                )

                # solve for the new demixing vector
                if J is None:
                    WV = W @ V
                else:
                    WV = torch.cat(
                        (W @ V, J @ V[..., :n_src, :] - V[..., n_src:, :]), dim=-2
                    )

                # the new filter, unscaled
                new_w = solve_loaded_general(
                    WV, evec[..., [k]], load=eps, eps=eps
                ).conj()
                new_w = new_w[..., 0]

                # resolve scale
                scale = torch.einsum("...c,...d,...cd->...", new_w, new_w.conj(), V)
                scale = scale.real
                new_w = new_w / torch.sqrt(torch.clamp(scale[..., None], min=eps))
                new_w = new_w[..., None, :]

                # re-build the demixing matrix
                W = torch.cat([W[..., :k, :], new_w, W[..., k + 1 : n_src, :]], dim=-2)

            # demix
            Y = torch.einsum("...fcd,...dfn->...cfn", W, X)

            if verbose:
                print(cost(model, Y, W, J, g))

        if proj_back_mic is not None:
            Y, a = projection_back_from_demixing_matrix(
                Y, W, J=J, ref_mic=proj_back_mic, load=eps
            )

        return Y
