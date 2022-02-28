from typing import List, Optional

import torch as pt

from .base import SourceModelBase
from .linalg import bmm, divide, eigh, hermite, inv_2x2, mag, mag_sq, multiply
from .models import LaplaceModel
from .parameters import eps_models


def control_scale(X):
    # Here we control the scale of X
    g = pt.sqrt(1e-5 * pt.mean(mag_sq(X), dim=(-2, -1), keepdim=True))
    X = divide(X, g, eps=1e-5)
    X = pt.view_as_complex(pt.view_as_real(X) / pt.clamp(g[..., None], min=1e-5))
    return X


def divide(num, denom, eps=1e-7):
    return pt.view_as_complex(
        pt.view_as_real(num) / pt.clamp(denom[..., None], min=eps)
    )


def spatial_model_update_iss(
    X: pt.Tensor,
    weights: pt.Tensor,
    W: Optional[pt.Tensor] = None,
    A: Optional[pt.Tensor] = None,
):
    """
    Apply the spatial model update via the iterative source steering rules

    Parameters
    ----------
    X: torch.Tensor, shape (..., n_channels, n_frequencies, n_frames)
        The input signal
    weights: torch.Tensor, shape (..., n_channels, n_frequencies, n_frames)
        The weights obtained from the source model to compute
        the weighted statistics
    W: torch.Tensor, shape (..., n_frequencies, n_channels, n_channels), optional
        The demixing matrix, it is updated if provided
    A: torch.Tensor, shape (..., n_frequencies, n_channels, n_channels), optional
        The mixing matrix, it is updated if provided

    Returns
    -------
    X: torch.Tensor, shape (n_frequencies, n_channels, n_frames)
        The updated source estimates
    """
    n_chan, n_freq, n_frames = X.shape[-3:]

    # Update now the demixing matrix
    for s in range(n_chan):
        Xr = multiply(weights, X[..., None, s, :, :])
        v_num = pt.sum(X * Xr.conj(), dim=-1) / n_frames
        v_denom = pt.sum(weights * mag_sq(X[..., None, s, :, :]), dim=-1) / n_frames

        v = divide(v_num, v_denom, eps=1e-3)
        v_s = 1.0 - (1.0 / pt.sqrt(pt.clamp(v_denom[..., s, :], min=1e-3)))
        v[..., s, :] = v_s

        # update demixed signals
        X = X - v[..., None] * X[..., s, None, :, :]

        if W is not None:
            vloc = v.transpose(-2, -1)
            Ws = W[..., s, :]
            W = W - vloc[..., None] * Ws[..., None, :]

        if A is not None:
            vloc = v.transpose(-2, -1)
            u = divide(bmm(A, vloc[..., None]), 1.0 - v_s[..., None, None])

            # we need to correct this when complex will be better supported
            # As = A[..., :, s]
            # A[..., :, s] = As + u[..., 0]
            canon_basis_vec = u.new_zeros(n_chan)
            canon_basis_vec[s] = 1.0
            A = A + u * canon_basis_vec

    if W is None and A is None:
        return X
    elif W is not None and A is None:
        return X, W
    elif W is None and A is not None:
        return X, A
    else:
        return X, W, A


def spatial_model_update_ip2(Xo: pt.Tensor, weights: pt.Tensor):
    """
    Apply the spatial model update via the generalized eigenvalue decomposition.
    This method is specialized for two channels.

    Parameters
    ----------
    Xo: torch.Tensor, shape (..., n_frequencies, n_channels, n_frames)
        The microphone input signal with n_chan == 2
    weights: torch.Tensor, shape (..., n_frequencies, n_channels, n_frames)
        The weights obtained from the source model to compute
        the weighted statistics

    Returns
    -------
    X: torch.Tensor, shape (n_frequencies, n_channels, n_frames)
        The updated source estimates
    """
    assert Xo.shape[-3] == 2, "This method is specialized for two channels processing."

    V = []
    for k in [0, 1]:
        # shape: (n_batch, n_freq, n_chan, n_chan)
        Vloc = pt.einsum(
            "...fn,...cfn,...dfn->...fcd", weights[..., k, :, :], Xo, Xo.conj()
        )
        Vloc = Vloc / Xo.shape[-1]
        # make sure V is hermitian symmetric
        Vloc = 0.5 * (Vloc + hermite(Vloc))
        V.append(Vloc)

    eigval, eigvec = eigh(V[1], V[0], eps=1e-7)

    # reverse order of eigenvectors
    eigvec = pt.flip(eigvec, dims=(-1,))

    for k in [0, 1]:
        scale = abs(pt.conj(eigvec[..., None, :, k]) @ (V[k] @ eigvec[..., :, None, k]))
        eigvec[..., :, k : k + 1] = divide(
            eigvec[..., :, k : k + 1],
            pt.sqrt(pt.clamp(scale, min=1e-7)),
            eps=1e-7,
        )

    # W = hermite(eigvec)
    # A = inv_2x2(W)

    # X = bmm(hermite(eigvec), Xo.transpose(-3, -2)).transpose(-3, -2)
    X = pt.einsum("...fcd,...dfn->...cfn", hermite(eigvec), Xo)

    return X  # , W, A


def auxiva_iss(
    X: pt.Tensor,
    n_iter: Optional[int] = 20,
    model: Optional[callable] = None,
    eps: Optional[float] = None,
    two_chan_ip2: Optional[bool] = False,
    proj_back: Optional[bool] = False,
    checkpoints_iter: Optional[List[int]] = None,
    checkpoints_list: Optional[List] = None,
) -> pt.Tensor:

    """
    Blind source separation based on independent vector analysis with
    alternating updates of the mixing vectors

    Parameters
    ----------
    X: Tensor, shape (..., n_channels, n_frequencies, n_frames)
        STFT representation of the signal
    n_iter: int, optional
        The number of iterations (default 20)
    model: SourceModel
        The model of source distribution (default: Laplace)

    Returns
    -------
    X: Tensor, shape (..., n_channels, n_frequencies, n_frames)
        STFT representation of the signal after separation
    """

    n_chan, n_freq, n_frames = X.shape[-3:]

    if eps is None:
        eps = eps_models["laplace"]

    if model is None:
        model = LaplaceModel()

    # for now, only supports determined case
    assert callable(model)

    if n_chan == 2 and two_chan_ip2:
        Xo = X

    if proj_back:
        W = X.new_zeros((n_freq, n_chan, n_chan))
        A = X.new_zeros((n_freq, n_chan, n_chan))
        W[:] = pt.eye(n_chan).type_as(W)

    for epoch in range(n_iter):

        if checkpoints_iter is not None and epoch in checkpoints_iter:
            checkpoints_list.append(X)

        # shape: (n_chan, n_freq, n_frames)
        # model takes as input a tensor of shape (..., n_frequencies, n_frames)
        weights = model(X)

        # we normalize the sources to have source to have unit variance prior to
        # computing the model
        g = pt.clamp(pt.mean(mag_sq(X), dim=(-2, -1), keepdim=True), min=1e-5)
        X = divide(X, pt.sqrt(g))
        weights = weights * g

        if n_chan == 2 and two_chan_ip2:
            # Here are the exact/fast updates for two channels using the GEVD
            X = spatial_model_update_ip2(Xo, weights)

        else:
            # Iterative Source Steering updates
            X = spatial_model_update_iss(X, weights)

    return X
