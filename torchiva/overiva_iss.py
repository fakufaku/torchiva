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
from torch.utils.checkpoint import CheckpointFunction

from .linalg import divide, hankel_view, hermite, mag_sq, multiply, solve_loaded
from .models import LaplaceModel
from .parameters import eps_models
from .auxiva_t_iss import projection_back_from_input


def reordered_eye(n_dim, n_freq, batch_shape):
    eye = torch.eye(n_dim)
    eye = torch.broadcast_to(eye[:, None, :], batch_shape + (n_dim, n_freq, n_dim))
    return eye


def check_inv_pair(A, B):
    prod = torch.einsum("...sfc,...cfd->...fsd", A, B)
    error = abs(prod - torch.eye(B.shape[-1]).type_as(prod)).mean()
    return error


def make_struct_inv_upper_left(W, J):
    n_src, n_freq, n_chan = W.shape[-3:]

    if n_src < n_chan:
        W1 = W[..., :, :, :n_src]
        W2 = W[..., :, :, n_src:]
        B_inv = W1 + torch.einsum("...sfc,...cfd->...sfd", W2, J)
    elif n_src == n_chan:
        B_inv = W
    else:
        raise NotImplementedError()
    return B_inv


def check_struct_inv_pair(W, J, B):
    B_inv = make_struct_inv_upper_left(W, J)
    return check_inv_pair(B, B_inv)


def make_demix(W, J):
    if J is not None:
        batch_shape = W.shape[:-3]
        n_src, n_freq, n_chan = W.shape[-3:]
        n_r = n_chan - n_src
        eye = torch.eye(J.shape[-3]).type_as(J)
        eye = reordered_eye(n_r, n_freq, batch_shape).type_as(W)
        return torch.cat((W, torch.cat((J, -eye), dim=-1)), dim=-3)
    else:
        return W


def check_mix_demix(W, J, A):
    if J is None:
        return check_inv_pair(W, A)

    else:
        return check_inv_pair(make_demix(W, J), A)


def reordered_inv(A):
    return torch.linalg.inv(A.transpose(-3, -2)).transpose(-2, -3)


def reordered_solve(A, b):
    sol = torch.linalg.solve(A.transpose(-3, -2), b.transpose(-3, -2))
    return sol.transpose(-2, -3)


def compute_cost(model, W, J, Y, Z):
    target_cost = model.cost(Y)

    WW = make_demix(W, J)
    _, ld = torch.linalg.slogdet(WW.transpose(-3, -2))
    logdet_cost = -2.0 * ld.sum(dim=-1)

    cost = target_cost + logdet_cost

    return cost


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


def demix_inv(W, J):

    n_src, n_freq, n_chan = W.shape[-3:]

    if J is None and n_src == n_chan:
        return reordered_inv(W)

    elif J is not None:

        # upper left block
        W1, W2 = W[..., :, :, :n_src], W[..., :, :, n_src:]
        B = torch.einsum("...sfc,...cfk->...sfk", W2, J)
        B = B + W1
        B = reordered_inv(B)

        # lower left block
        JB = torch.einsum("...cfs,...sfd->...cfd", J, B)

        # upper right block
        BW2 = torch.einsum("...dfs,...sfc->...dfc", B, W2)

        # lower right block
        JBW2 = torch.einsum("...cfd,...dfe->...cfe", JB, W2)
        JBW2 = JBW2 - reordered_eye(n_chan - n_src, n_freq, W.shape[:-3]).type_as(JBW2)

        A = torch.cat(
            (torch.cat((B, BW2), dim=-1), torch.cat((JB, JBW2), dim=-1)), dim=-3
        )

        return A

    else:
        raise ValueError("Incompatible arguments")


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

    # no need to backprop through diagonal loading
    with torch.no_grad():
        load = abs(torch.diagonal(mat, dim1=-2, dim2=-1)).sum(dim=-1)
        load = load[..., None, None]
        load = torch.clamp(load * eps, min=eps)
        load = load * torch.broadcast_to(torch.eye(n_src).type_as(mat), mat.shape)

    J_H = torch.linalg.solve(
        A[..., :n_src] + load, A[..., n_src:]
    )
    J = J_H.conj().moveaxis(-1, -3)  # (..., n_chan - n_src, n_freq, n_src)

    return J


def overiss2_compute_background_update_vector(p, d, g, eps=1e-3):

    gdg = torch.sum((g.real ** 2 + g.imag ** 2) / (d + eps), dim=-2)
    gdp = torch.sum(g.conj() * p / (d + eps), dim=-2)

    b = 1 - gdp
    b1 = b.real ** 2 + b.imag ** 2
    a = b1 * gdg

    beta = (-b1 + torch.sqrt(b1 ** 2 + 4 * a)) / (2.0 * a)

    ell1 = beta * b
    ell2 = 1.0 / torch.sqrt(eps + gdg)
    ell = torch.where(b.real ** 2 + b.imag ** 2 > eps ** 2, ell1, ell2.type_as(ell1))

    v = (p - ell[..., None, :] * g) / (d + eps)

    return v


def overiss2_background_update(
    bak_src: int,
    Y: torch.Tensor,  # targets
    Z: torch.Tensor,  # background
    X: torch.Tensor,  # input
    W: torch.Tensor,  # demix matrix
    J: torch.Tensor,  # background demixing matrix
    A: torch.Tensor,  # mixing matrix
    weights: torch.Tensor,
    eps: Optional[float] = 1e-3,
) -> torch.Tensor:
    """
    ISS style update for the background part in overdet ISS
    """

    n_src, n_freq, n_frames = Y.shape[-3:]
    n_chan = X.shape[-3]

    Zs = Z[..., bak_src, :, :] + X[..., n_src + bak_src, :, :]
    norm = 1.0 / n_frames

    # pad weights with ones for extra channels
    ones = weights.new_ones(weights.shape[:-3] + (n_chan - n_src, n_freq, n_frames))
    weights = torch.cat((weights, ones), dim=-3)

    YZ = torch.cat((Y, Z), dim=-3)
    p_vec = torch.einsum("...cfn,...cfn,...fn->...cf", YZ, weights, Zs.conj())
    p_vec = p_vec * norm

    d_vec = torch.einsum("...cfn,...fn->...cf", weights, Zs.real ** 2 + Zs.imag ** 2)
    d_vec = d_vec * norm

    # form top of mixing matrix
    A2 = torch.einsum("...sfc,...cfd->...sfd", A, W[..., :, :, n_src:])
    A2 = torch.cat((A, A2), dim=-1)
    g_vec = torch.einsum("...fs,...sfc->...cf", J[..., bak_src, :, :], A2)
    g_vec = g_vec.conj()

    v = overiss2_compute_background_update_vector(p_vec, d_vec, g_vec, eps=eps)

    YZ = YZ - torch.einsum("...cf,...fn->...cfn", v, Zs)

    return v, YZ[..., :n_src, :, :], YZ[..., n_src:, :, :]

    # v[..., :n_src, :] = 0.0
    # return v, Y, YZ[..., n_src:, :, :]


def mixing_update_type2(
    A: torch.Tensor,  # (..., n_chan, n_freq, n_chan)
    v: torch.Tensor,  # (..., n_chan, n_freq)
    g: torch.Tensor,  # (..., n_freq, n_src)
    eps: Optional[float] = 1e-15,
) -> torch.Tensor:

    # A = A.type(torch.complex128)
    # v = v.type(torch.complex128)
    # g = g.type(torch.complex128)

    Av = torch.einsum("...sfc,...cf->...sf", A[..., : v.shape[-2]], v)
    gA = torch.einsum("...fs,...sfc->...fc", g, A[..., : g.shape[-1], :, :])
    denom = 1.0 - torch.einsum("...cf,...fc->...f", v, gA)

    B = torch.einsum("...sf,...fc->...sfc", Av, gA / (self.eps + denom[..., :, None]))

    A = A + B

    # return A.type(torch.complex64)
    return A


def matrices_update_type1(
    W: torch.Tensor,
    H: torch.Tensor,
    A: torch.Tensor,
    v: torch.Tensor,
    src: int,
    eps: Optional[float] = 1e-3,
) -> torch.Tensor:

    W = W - torch.einsum("...cf,...fd->...cfd", v, W[..., src, :, :])
    H = H - torch.einsum("...cf,...fdt->...cfdt", v, H[..., src, :, :, :])

    # mixing matrix
    u = torch.einsum("...sfc,...cf->...sf", A[..., : v.shape[-2]], v)
    denom = 1.0 - v[..., [src], :]
    u = u / (denom + eps)

    A = A.clone()
    A[..., src] = A[..., src] + u

    return W, H, A


def matrices_update_type2(
    W: torch.Tensor,  # (..., n_src, n_freq, n_chan)
    J: torch.Tensor,
    A: torch.Tensor,  # (..., n_chan, n_freq, n_chan)
    v: torch.Tensor,  # (..., n_chan, n_freq)
    g: torch.Tensor,  # (..., n_freq, n_src)
    eps: Optional[float] = 1e-3,
) -> torch.Tensor:

    W = W.type(torch.complex128)
    J = J.type(torch.complex128)
    A = A.type(torch.complex128)
    v = v.type(torch.complex128)
    g = g.type(torch.complex128)

    n_src, n_freq, n_chan = W.shape[-3:]

    v2 = v[..., :n_src, :] + torch.einsum(
        "...sfc,...cf->...sf", W[..., :, n_src:], v[..., n_src:, :]
    )

    # demixing matrix update
    W1 = W[..., :n_src] - torch.einsum("...cf,...fd->...cfd", v[..., :n_src, :], g)
    W = torch.cat((W1, W[..., n_src:]), dim=-1)

    # overdet. demix matrix
    J = J - torch.einsum("...cf,...fd->...cfd", v[..., n_src:, :], g)

    Av = torch.einsum("...sfc,...cf->...sf", A[..., : v2.shape[-2]], v2)
    gA = torch.einsum("...fs,...sfc->...fc", g, A[..., : g.shape[-1], :, :])
    denom = 1.0 - torch.einsum("...cf,...fc->...f", v2, gA)

    mul = denom.conj() / (mag_sq(denom) + eps)

    B = torch.einsum("...sf,...fc->...sfc", Av, gA * mul[..., None])

    A = A + B

    # return W, J, A
    return W.type(torch.complex64), J.type(torch.complex64), A.type(torch.complex64)


def overiss2_updates_with_H(
    Y: torch.Tensor,
    X: torch.Tensor,
    X_bar: torch.Tensor,
    W: torch.Tensor,
    H: torch.Tensor,
    A: torch.Tensor,  # A = W ** (-1)
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
    n_src, n_freq, n_frames = Y.shape[-3:]
    if J is None:
        n_chan = n_src
    else:
        n_chan = n_src + Z.shape[-3]
    n_taps = X_bar.shape[-2]

    # we make a copy because we need to do some inplace operations
    H = H.clone()
    W = W.clone()

    # background part
    if Z is not None and J is not None:
        for bak in range(n_chan - n_src):
            v, Y, Z = overiss2_background_update(
                bak, Y, Z, X, W, J, A, weights, eps=eps
            )
            g = J[..., bak, :, :]
            W, J, A = matrices_update_type2(W, J, A, v, g, eps=eps)

    # source separation part
    for src in range(n_src):
        v = iss_block_update_type_1(src, Y, weights, eps=eps)
        Y = Y - torch.einsum("...cf,...fn->...cfn", v, Y[..., src, :, :])
        W, H, A = matrices_update_type1(W, H, A, v, src, eps=eps)

    # background part
    if Z is not None and J is not None:
        for bak in range(n_chan - n_src):
            v = iss_block_update_type_2(Y, Z[..., bak, :, :], weights, eps=eps)
            Y = Y - torch.einsum("...cf,...fn->...cfn", v, Z[..., bak, :, :])
            W[..., :n_src] = W[..., :n_src] - torch.einsum(
                "...cf,...fd->...cfd", v, J[..., bak, :, :]
            )
            W[..., n_src + bak] = W[..., n_src + bak] + v

    # dereverberation part
    for src in range(n_chan):
        for tap in range(n_taps):
            v = iss_block_update_type_3(src, tap, Y, X_bar, weights, eps=eps)
            Y = Y - torch.einsum("...cf,...fn->...cfn", v, X_bar[..., src, :, tap, :])
            HV = H[..., src, tap] + v
            H[..., src, tap] = HV

    return Y, Z, W, H, J, A


def rescale(Y, W, H, A=None, eps=1e-5):
    # we normalize the sources to have source to have unit variance prior to
    # computing the model
    g = torch.clamp(torch.mean(mag_sq(Y), dim=(-2, -1), keepdim=True), min=eps)
    g_sqrt = torch.sqrt(g)
    Y = Y / torch.clamp(g_sqrt, min=eps)
    W = W / torch.clamp(g_sqrt, min=eps)
    H = H / torch.clamp(g_sqrt[..., None], min=eps)
    if A is not None:
        A = A * g_sqrt.transpose(-3, -1)

    return Y, W, H, A, g


def overiss2_one_iter(Y, Z, X, X_bar, C_XX, C_XbarX, W, H, J, A, model, eps=1e-3):

    # Y, W, H, A, g = rescale(Y, W, H, A, eps)

    # shape: (n_chan, n_freq, n_frames)
    # model takes as input a tensor of shape (..., n_frequencies, n_frames)
    weights = model(Y)

    # <---
    # we normalize the sources to have source to have unit variance prior to
    # computing the model
    g = torch.clamp(
        torch.mean(mag_sq(Y), dim=(-2, -1), keepdim=True), min=eps
    )
    g_sqrt = torch.sqrt(g)
    Y = divide(Y, g_sqrt, eps=eps)
    W = divide(W, g_sqrt, eps=eps)
    A = A * g_sqrt.transpose(-3, -1)
    H = divide(H, g_sqrt[..., None], eps=eps)
    weights = weights * g
    g = torch.Tensor([1.0])  # for cost computation
    # <---

    # Iterative Source Steering updates
    Y, Z, W, H, J, A = overiss2_updates_with_H(
        Y, X, X_bar, W, H, A, weights, J=J, Z=Z, eps=eps
    )

    return Y, Z, W, H, J, A, g


def overiss2_one_iter_dmc(X, X_bar, C_XX, C_XbarX, W, H, J, A, model, eps=1e-3):
    Y = demix_derev(X, X_bar, W, H)
    Z = demix_background(X, J)
    Y, Z, W, H, J, A, g = overiss2_one_iter(
        Y, Z, X, X_bar, C_XX, C_XbarX, W, H, J, A, model, eps
    )
    return W, H, J, A, g


def over_iss_t_one_iter(Y, X, X_bar, C_XX, C_XbarX, W, H, J, model, eps=1e-3):

    # Y, W, H, _, g = rescale(Y, W, H, None, eps)

    # shape: (n_chan, n_freq, n_frames)
    # model takes as input a tensor of shape (..., n_frequencies, n_frames)
    weights = model(Y)

    # <---
    # we normalize the sources to have source to have unit variance prior to
    # computing the model
    g = torch.clamp(
        torch.mean(mag_sq(Y), dim=(-2, -1), keepdim=True), min=eps
    )
    g_sqrt = torch.sqrt(g)
    Y = divide(Y, g_sqrt, eps=eps)
    W = divide(W, g_sqrt, eps=eps)
    H = divide(H, g_sqrt[..., None], eps=eps)
    weights = weights * g
    g = torch.Tensor([1.0])  # for cost computation
    # <---

    # we normalize the sources to have source to have unit variance prior to
    # Update the background part
    if J is not None:
        J = background_update(W, H, C_XX, C_XbarX, eps=eps)
        Z = demix_background(X, J)  # Z is None if J is None
    else:
        Z = None

    # Iterative Source Steering updates
    Y, W, H = iss_updates_with_H(Y, X_bar, W, H, weights, Z=Z, J=J, eps=eps)

    return Y, W, H, J, g


def over_iss_t_one_iter_dmc(X, X_bar, C_XX, C_XbarX, W, H, J, model, eps=1e-3, *model_params):

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
        verbose: Optional[bool] = False,
    ):
        super().__init__()

        self.n_taps = n_taps
        self.n_delay = n_delay
        self.n_iter = n_iter
        self.proj_back = proj_back
        self.ref_mic = ref_mic
        self.use_dmc = use_dmc
        self.verbose = verbose

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
                f"Underdetermined source separation (n_src={n_src},"
                f" n_channels={n_chan}) is not supported"
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
        eye = torch.eye(n_src, n_chan).type_as(W)
        W[...] = eye[:, None, :]

        H = X.new_zeros(batch_shape + (n_src, n_freq, n_chan, self.n_taps))

        if is_overdet:
            J = background_update(W, H, C_XX, C_XbarX, eps=self.eps)
        else:
            J = None

        Y = demix_derev(X, X_bar, W, H)

        rescaled_cost = 0.0

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
                model_params = [p for p in self.model.parameters()]
                W, H, J, g = torch_checkpoint(
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
                    *model_params,
                    preserve_rng_state=True,
                )
            else:
                Y, W, H, J, g = over_iss_t_one_iter(
                    Y, X, X_bar, C_XX, C_XbarX, W, H, J, self.model, eps=self.eps,
                )

            # keep track of rescaling cost
            rescaled_cost = rescaled_cost - n_freq * torch.sum(torch.log(g))

            if self.verbose and hasattr(self.model, "cost"):
                Z = demix_background(X, J)
                cost = compute_cost(self.model, W, J, Y, Z) + rescaled_cost
                print(f"Epoch {epoch}: {cost.sum()}")

        if use_dmc:
            # when using DMC, we have not yet computed Y explicitely
            Y = demix_derev(X, X_bar, W, H)

        # projection back
        if proj_back:
            # projection back from the input signal directly
            """
            Z = demix_background(X, J)
            if Z is not None:
                Y2 = torch.cat((Y, Z), dim=-3)
            else:
                Y2 = Y
            Y, *_ = projection_back_from_input(Y2, X, X_bar, ref_mic=self.ref_mic, eps=self.eps)
            """
            # projection back by inverting the demixing matrix
            a = projection_back_weights(W, J=J, eps=self.eps, ref_mic=self.ref_mic)
            Y = a * Y

        self.W = W
        self.J = J

        return Y


class OverISS_T_2(torch.nn.Module):
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
        verbose: Optional[bool] = False,
    ):
        super().__init__()

        self.n_taps = n_taps
        self.n_delay = n_delay
        self.n_iter = n_iter
        self.proj_back = proj_back
        self.ref_mic = ref_mic
        self.use_dmc = use_dmc
        self.verbose = verbose

        # the different parts of the demixing matrix
        self.W = None  # target sources
        self.H = None  # reverb for target sources
        self.J = None  # background in overdetermined case
        self.A = None  # mixing matrix

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
                f"Underdetermined source separation (n_src={n_src},"
                f" n_channels={n_chan}) is not supported"
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
        eye = torch.eye(n_src, n_chan).type_as(W)
        W[...] = eye[:, None, :]

        H = X.new_zeros(batch_shape + (n_src, n_freq, n_chan, self.n_taps))

        if is_overdet:
            J = background_update(W, H, C_XX, C_XbarX, eps=self.eps)
        else:
            J = None

        A = reordered_inv(make_struct_inv_upper_left(W, J))

        Y = demix_derev(X, X_bar, W, H)
        Z = demix_background(X, J)

        if self.verbose and hasattr(self.model, "cost"):
            cost = compute_cost(self.model, W, J, Y, Z)
            print(f"Epoch {-1}: {cost}")

        rescaled_cost = 0.0

        for epoch in range(n_iter):

            if self.checkpoints_iter is not None and epoch in self.checkpoints_iter:
                # self.checkpoints_list.append(X)
                if epoch == 0:
                    checkpoints_list.append(Y.clone().detach())
                else:
                    a = A[..., [self.ref_mic], :, :n_src].transpose(-3, -1)
                    checkpoints_list.append((Y * a).detach())

            if use_dmc:
                W, H, J, A, g = torch_checkpoint(
                    overiss2_one_iter_dmc,
                    X,
                    X_bar,
                    C_XX,
                    C_XbarX,
                    W,
                    H,
                    J,
                    A,
                    self.model,
                    self.eps,
                    preserve_rng_state=True,
                )
            else:
                Y, Z, W, H, J, A, g = overiss2_one_iter(
                    Y, Z, X, X_bar, C_XX, C_XbarX, W, H, J, A, self.model, eps=self.eps
                )

            # keep track of rescaling cost
            rescaled_cost = rescaled_cost - n_freq * torch.sum(torch.log(g))

            if self.verbose and hasattr(self.model, "cost"):
                cost = compute_cost(self.model, W, J, Y, Z) + rescaled_cost
                print(f"Epoch {epoch}: {cost.sum()}")

        if self.verbose:
            print("Check matrix inverse", check_struct_inv_pair(W, J, A))

        # projection back
        if proj_back:
            if self.ref_mic > n_src:
                raise NotImplementedError(
                    "Reference mic should be less than number of sources"
                )
            a = A[..., [self.ref_mic], :, :n_src].transpose(-3, -1)
            # a = projection_back_weights(W, J=J, eps=self.eps, ref_mic=self.ref_mic)
            if use_dmc:
                Y = a * demix_derev(X, X_bar, W, H)
            else:
                Y = a * Y

        return Y
