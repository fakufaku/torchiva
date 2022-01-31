import math
from typing import List, Optional

import torch as pt

from .base import SourceModelBase
from .linalg import bmm, divide, eigh, hermite, mag, mag_sq, multiply, inv_loaded
from .models import LaplaceModel
from .parameters import eps_models


def normalize(x):
    return x / pt.linalg.norm(x, dim=-2, keepdim=True)


def smallest_eigenvector_eigh_cpu(V):
    dev = V.device

    # diagonal loading factor
    # dload = 1e-5 * pt.diag(pt.arange(V.shape[-1], device=V.device))
    # V = (V + dload).cpu()
    V = V.cpu()
    r = pt.linalg.eigh(V)

    ev = r.eigenvalues[..., 0].to(dev)
    ev = pt.reciprocal(pt.clamp(ev, min=1e-5))

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
        x = pt.einsum("...fcd,...fd->...fc", V_inv, x)
        ev = pt.linalg.norm(x, dim=-1)
        x = x / ev[..., None]

    # fix sign of first element to positive
    s = pt.conj(pt.sgn(x[..., 0]))
    x = x * s[..., None]

    return ev, x


def adjust_global_scale(Y, ref):
    num = pt.mean(pt.conj(Y) * ref, dim=(-2, -1), keepdim=True)
    denom = pt.mean(Y.abs().square(), dim=(-2, -1), keepdim=True)
    denom = pt.clamp(denom, min=1e-7)
    scale = num / denom
    return Y * scale


def five(
    X: pt.Tensor,
    n_iter: Optional[int] = 20,
    model: Optional[SourceModelBase] = None,
    eps: Optional[float] = None,
    ref_mic: Optional[float] = None,
    use_wiener: Optional[bool] = True,
    use_n_power_iter: Optional[int] = None,
    checkpoints_iter: Optional[List[int]] = None,
    checkpoints_list: Optional[List] = None,
) -> pt.Tensor:

    """
    Blind source extraction based on FIVE (fast independent vector extraction)

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
    X: Tensor, shape (..., n_frequencies, n_frames)
        STFT representation of the signal after extraction
    """

    n_chan, n_freq, n_frames = X.shape[-3:]

    if eps is None:
        eps = eps_models["laplace"]

    if model is None:
        model = LaplaceModel()

    # for now, only supports determined case
    assert callable(model)

    # remove DC part
    X = X[..., 1:, :]

    # Pre-whitening

    # covariance matrix of input signal (n_freq, n_chan, n_chan)
    Cx = pt.einsum("...cfn,...dfn->...fcd", X, X.conj()) / n_frames
    Cx = 0.5 * (Cx + hermite(Cx))

    # We will need the inverse square root of Cx
    # e_val, e_vec = pt.linalg.eigh(Cx)
    e_val, e_vec = eigh(Cx)

    # put the eigenvalues in descending order
    e_vec = pt.flip(e_vec, dims=[-1])
    e_val = pt.flip(e_val, dims=[-1])

    # compute the whitening matrix
    e_val_sqrt = pt.sqrt(pt.clamp(e_val, min=1e-5))
    Q = pt.reciprocal(e_val_sqrt[..., :, None]) * hermite(e_vec)
    Qinv = e_vec * e_val_sqrt[..., None, :]

    # we keep the row of Q^{-1} corresponding to the reference mic for the
    # normalization
    Qinv_ref = Qinv[..., ref_mic, :]
    # Qinv_ref = e_vec[..., ref_mic, None, :] * e_val_sqrt[..., None, :]

    # keep the input for scaling
    X_in = X

    # The whitened input signal
    X = pt.einsum("...fcd,...dfn->...cfn", Q, X)

    # Pick the initial signal as the largest component of PCA,
    # i.e., the first channel (because we flipped the order above)
    Y = X[..., 0, :, :]

    # Y = adjust_global_scale(Y, X_in[..., ref_mic, None, :, :])

    # projection back
    if ref_mic is not None:
        z = Q[..., 0, None, :] @ Cx[..., ref_mic, None]
        z = z.transpose(-3, -2)
        Y = z[..., 0, :, :] * Y

    for epoch in range(n_iter):

        if checkpoints_iter is not None and epoch in checkpoints_iter:
            checkpoints_list.append(X)

        # shape: (n_chan, n_freq, n_frames)
        # model takes as input a tensor of shape (..., n_src, n_masks, n_frequencies, n_frames)
        weights = model(Y)

        # compute the weighted spatial covariance matrix
        V = pt.einsum("...fn,...cfn,...dfn->...fcd", weights, X, X.conj())
        V = 0.5 * (V + hermite(V))

        # now compute the demixing vector
        # inv_eigenvalue (..., n_freq)
        # eigenvector (..., n_freq, n_chan)
        if use_n_power_iter is None:
            inv_eigenvalue, eigenvector = smallest_eigenvector_eigh_cpu(V)
        else:
            inv_eigenvalue, eigenvector = smallest_eigenvector_power_method(V, n_iter=use_n_power_iter)

        # projection back
        z = pt.einsum("...fc,...fc->...f", Qinv_ref, eigenvector)
        w = eigenvector * pt.conj(z[..., None])

        # Wiener filter
        if use_wiener:
            w = w * (1.0 - 1.0 / inv_eigenvalue[..., None])

        # the new output
        Y = pt.einsum("...fc,...cfn->...fn", w.conj(), X)

    # add back DC offset
    pad_shape = Y.shape[:-2] + (1,) + Y.shape[-1:]
    Y = pt.cat((Y.new_zeros(pad_shape), Y), dim=-2)

    return Y[..., None, :, :]
