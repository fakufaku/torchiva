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

from typing import List, NoReturn, Optional, Tuple

import torch
from torch.utils.checkpoint import CheckpointFunction
from torch.utils.checkpoint import checkpoint as torch_checkpoint

from .base import DRBSSBase
from .linalg import (
    divide,
    hankel_view,
    hermite,
    mag_sq,
    multiply,
    solve_loaded,
    solve_loaded_general,
)


def demix_derev(X, X_bar, W, H):
    reverb = torch.einsum("...cfdt,...dftn->...cfn", H, X_bar)
    sep = torch.einsum("...cfd,...dfn->...cfn", W, X)
    return sep - reverb


def demix_background(X, J):
    if J is not None:
        n_src = J.shape[-1]
        return (
            torch.einsum("...cfd, ...dfn->...cfn", J, X[..., :n_src, :, :])
            - X[..., n_src:, :, :]
        )
    else:
        return None


def iss_block_update_type_1(
    src: int,
    X: torch.Tensor,
    weights: torch.Tensor,
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


def background_update(W, H, C_XX, C_XbarX, eps=1e-5):
    """
    Recomputes J based on W, H, and C_XX = E[X X^H] and C_XbarX = E[ X_bar X^H ]
    """
    n_src, n_freq, n_chan = W.shape[-3:]

    A1 = torch.einsum("...sfc,...cfd->...fsd", W, C_XX)
    A2 = torch.einsum("...sfdt,...dtfc->...fsc", H, C_XbarX)
    A = A1 + A2  # (..., n_freq, n_src, n_chan)

    mat = A[..., :n_src]
    rhs = A[..., n_src:]

    with torch.no_grad():
        # normalize the rows of A without changing the solution
        # for numerical stability
        norm = torch.linalg.norm(mat.detach(), dim=-1, keepdim=True)
        weights = 1.0 / torch.clamp(norm, min=eps)

    load = eps * torch.eye(n_src).type_as(mat)

    # make eigenvalues positive and scale
    mat2 = mat * weights
    rhs = rhs * weights
    mat = torch.einsum("...km,...kn->...mn", mat2.conj(), mat2)
    rhs = torch.einsum("...km,...kn->...mn", mat2.conj(), rhs)

    J_H = torch.linalg.solve(mat + load, rhs)
    J = J_H.conj().moveaxis(-1, -3)  # (..., n_chan - n_src, n_freq, n_src)

    return J


def over_iss_t_one_iter(Y, X, X_bar, C_XX, C_XbarX, W, H, J, model, eps=1e-3):

    # shape: (n_chan, n_freq, n_frames)
    # model takes as input a tensor of shape (..., n_frequencies, n_frames)
    weights = model(Y)

    # <---
    # we normalize the sources to have source to have unit variance prior to
    # computing the model
    g = torch.clamp(torch.mean(mag_sq(Y), dim=(-2, -1), keepdim=True), min=eps)
    g_sqrt = torch.sqrt(g)
    Y = divide(Y, g_sqrt, eps=eps)
    W = divide(W, g_sqrt, eps=eps)
    H = divide(H, g_sqrt[..., None], eps=eps)
    weights = weights * g
    # <---

    # we normalize the sources to have source to have unit variance prior to
    # Update the background part
    if J is not None:
        J = background_update(W, H, C_XX, C_XbarX, eps=1e-3)
    Z = demix_background(X, J)  # Z is None if J is None

    # Iterative Source Steering updates
    Y, W, H = iss_updates_with_H(Y, X_bar, W, H, weights, Z=Z, J=J, eps=eps)

    return Y, W, H, J, g


def over_iss_t_one_iter_dmc(
    X, X_bar, C_XX, C_XbarX, W, H, J, model, eps=1e-3, *model_params
):

    Y = demix_derev(X, X_bar, W, H)

    Y, W, H, J, g = over_iss_t_one_iter(
        Y, X, X_bar, C_XX, C_XbarX, W, H, J, model, eps=eps
    )

    return W, H, J, g


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

        lhs = torch.einsum("...sfc->...fcs", B)

    else:
        # determined case
        rhs = eye[..., :, [ref_mic]]
        WT = W.transpose(-3, -2)
        lhs = WT.transpose(-2, -1)

    a = solve_loaded_general(lhs, rhs)

    return a.transpose(-3, -2)


class T_ISS(DRBSSBase):
    """
    Joint dereverberation and separation with *time-decorrelation iterative source steering* (T-ISS) [1]_.

    Parameters can also be specified during a forward call.
    In this case, the forward argument is only used in that forward process and **does not rewrite class attributes**.

    Parameters
    ----------
    n_iter: int, optional
        The number of iterations. (default: ``10``)
    n_taps: int, optional
        The length of the dereverberation filter.
        If set to ``0``, this method works as the normal AuxIVA with ISS update [2]_ (default: ``0``).
    n_delay: int, optional
        The number of delay for dereverberation (default: ``0``).
    n_src: int, optional
        The number of sources to be separated.
        When ``n_src < n_chan``, a computationally cheaper variant (Over-T-ISS) [3]_ is used.
        If set to ``None``,  ``n_src`` is set to ``n_chan`` (default: ``None``)
    model: torch.nn.Module, optional
        The model of source distribution.
        Mask estimation neural network can also be used.
        If ``None``, spherical Laplace is used (default: ``None``).
    proj_back_mic: int, optional
        The reference mic index to perform projection back.
        If set to ``None``, projection back is not applied (default: ``0``).
    use_dmc: bool, optonal
        If set to ``True``, memory efficient Demixing Matrix Checkpointing (DMC) [4]_ is used to compute the gradient.
        It reduces the memory cost to that of a single iteration when training neural source model (default: ``False``).
    eps: float, optional
        A small constant to make divisions and the like numerically stable (default:``None``).


    Methods
    --------
    forward(n_iter=None, n_taps=None, n_delay=None, n_src=None, model=None, proj_back_mic=None, use_dmc=None, eps=None)

    Parameters
    ----------
    X: torch.Tensor
        The input mixture in STFT-domain,
        ``shape (..., n_chan, n_freq, n_frames)``

    Returns
    ----------
    Y: torch.Tensor, ``shape (..., n_src, n_freq, n_frames)``
        The separated and dereverberated signal in STFT-domain

    Note
    ----
    This class can handle various BSS methods with ISS update rule depending on the specified arguments:
        * IVA-ISS: ``n_taps=0, n_delay=0, n_chan==n_src, model=LaplaceMoldel() or GaussMoldel()``
        * ILRMA-ISS: ``n_taps=0, n_delay=0, n_chan==n_src, model=NMFModel()``
        * DNN-IVA-ISS: ``n_taps=0, n_delay=0, n_chan==n_src, model=*DNN*``
        * OverIVA-ISS: ``n_taps=0, n_delay=0, n_chan < n_src``
        * ILRMA-T-ISS [1]_ : ``n_taps>0, n_delay>0, n_chan==n_src, model=NMFMoldel()``
        * DNN-T-ISS [4]_ : ``n_taps>0, n_delay>0, n_chan==n_src, model=*DNN*``
        * Over-T-ISS [3]_ : ``n_taps>0, n_delay>0, n_chan > n_src``


    References
    ----------
    .. [1] T. Nakashima, R. Scheibler, M. Togami, and N. Ono,
        "Joint dereverberation and separation with iterative source steering",
        ICASSP, 2021, https://arxiv.org/pdf/2102.06322.pdf.

    .. [2] R. Scheibler, and N Ono,
        "Fast and stable blind source separation with rank-1 updates"
        ICASSP, 2021,

    .. [3] R. Scheibler, W. Zhang, X. Chang, S. Watanabe, and Y. Qian,
        "End-to-End Multi-speaker ASR with Independent Vector Analysis",
        arXiv preprint arXiv:2204.00218, 2022, https://arxiv.org/pdf/2204.00218.pdf.

    .. [4] K. Saijo, and R. Scheibler,
        "Independence-based Joint Speech Dereverberation and Separation with Neural Source Model",
        arXiv preprint arXiv:2110.06545, 2022, https://arxiv.org/pdf/2110.06545.pdf.

    """

    def __init__(
        self,
        n_iter: Optional[int] = 10,
        n_taps: Optional[int] = 0,
        n_delay: Optional[int] = 0,
        n_src: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
        proj_back_mic: Optional[int] = 0,
        use_dmc: Optional[bool] = False,
        eps: Optional[float] = None,
    ):

        super().__init__(
            n_iter,
            n_taps=n_taps,
            n_delay=n_delay,
            n_src=n_src,
            model=model,
            proj_back_mic=proj_back_mic,
            use_dmc=use_dmc,
            eps=eps,
        )

        # the different parts of the demixing matrix
        self.W = None  # target sources
        self.H = None  # reverb for target sources
        self.J = None  # background in overdetermined case

    def forward(
        self,
        X: torch.Tensor,
        n_iter: Optional[int] = None,
        n_taps: Optional[int] = None,
        n_delay: Optional[int] = None,
        n_src: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
        proj_back_mic: Optional[int] = None,
        use_dmc: Optional[bool] = None,
        eps: Optional[float] = None,
    ) -> torch.Tensor:

        batch_shape = X.shape[:-3]
        n_chan, n_freq, n_frames = X.shape[-3:]

        (
            n_iter,
            n_taps,
            n_delay,
            n_src,
            model,
            proj_back_mic,
            use_dmc,
            eps,
        ) = self._set_params(
            n_iter=n_iter,
            n_taps=n_taps,
            n_delay=n_delay,
            n_src=n_src,
            model=model,
            proj_back_mic=proj_back_mic,
            use_dmc=use_dmc,
            eps=eps,
        )

        if n_src is None:
            n_src = n_chan
        elif n_src > n_chan:
            raise ValueError(
                f"Underdetermined source separation (n_src={n_src},"
                f" n_channels={n_chan}) is not supported"
            )

        # initialize source model if NMF
        self._reset(model)

        is_overdet = n_src < n_chan

        # shape (..., n_chan, n_freq, n_taps + n_delay + 1, block_size)
        X_pad = torch.nn.functional.pad(X, (n_taps + n_delay, 0))
        X_hankel = hankel_view(X_pad, n_taps + n_delay + 1)
        X_bar = X_hankel[..., : -n_delay - 1, :]  # shape (c, f, t, b)

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
        eye = torch.eye(n_src, n_chan).type_as(W)
        W[...] = eye[:, None, :]

        H = X.new_zeros(batch_shape + (n_src, n_freq, n_chan, n_taps))

        if is_overdet:
            J = background_update(W, H, C_XX, C_XbarX, eps=eps)
        else:
            J = None

        Y = demix_derev(X, X_bar, W, H)

        for epoch in range(n_iter):

            if use_dmc:
                model_params = [p for p in model.parameters()]
                W, H, J, g = torch_checkpoint(
                    over_iss_t_one_iter_dmc,
                    X,
                    X_bar,
                    C_XX,
                    C_XbarX,
                    W,
                    H,
                    J,
                    model,
                    eps,
                    *model_params,
                    preserve_rng_state=True,
                )
            else:
                Y, W, H, J, g = over_iss_t_one_iter(
                    Y,
                    X,
                    X_bar,
                    C_XX,
                    C_XbarX,
                    W,
                    H,
                    J,
                    model,
                    eps=eps,
                )

        if use_dmc:
            # when using DMC, we have not yet computed Y explicitely
            Y = demix_derev(X, X_bar, W, H)

        # projection back
        if proj_back_mic is not None and n_iter > 0:
            # projection back by inverting the demixing matrix
            a = projection_back_weights(W, J=J, eps=eps, ref_mic=proj_back_mic)
            Y = a * Y

        self.W = W
        self.J = J

        return Y
