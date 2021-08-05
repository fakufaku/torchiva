import math
from typing import List, Optional, Callable

import torch as pt

from .linalg import bmm, divide, eigh, hermite, mag, mag_sq, multiply
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

    x = r.eigenvectors[..., :, 0, None].to(dev)

    return ev, x


def smallest_eigenvector_power_method(V, n_iter=10):

    assert V.shape[-2] == V.shape[-1]

    # initial point
    # x = V.mean(dim=-1, keepdim=True)
    x = V.new_zeros(V.shape[:-1] + (1,)).uniform_()
    x = normalize(x)

    # compute inverse of input matrix
    V_inv = pt.linalg.inv(V)

    for epoch in range(n_iter):
        x = V_inv @ x
        ev = pt.linalg.norm(x, dim=(-2, -1))
        x = x / ev[..., None, None]

    # fix sign of first element to positive
    s = pt.conj(pt.sgn(x[..., 0, 0]))
    x = x * s[..., None, None]

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
    model: Optional[Callable] = None,
    eps: Optional[float] = None,
    ref_mic: Optional[float] = None,
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

    def freq_wise_bmm(a, b):
        """
        Parameters
        ----------
        a: (..., n_freq, n_channels_1, n_channels_2)
            op 1
        b: (..., n_channels_2, n_freq, n_frames)
            op 2

        Returns
        -------
        The batch matrix multiplication along the frequency axis,
        then flips the axis back
        Tensor of shape (..., channels_1, n_freq, n_frames)
        """
        return (a @ b.transpose(-3, -2)).transpose(-3, -2)

    def batch_abH(a, b):
        """
        Parameters
        -# ---------
        a: (..., n_channels_1, n_freq, n_frames)
            op 1
        b: (..., n_channels_2, n_freq, n_frames)
            op 2

        Returns
        -------
        Tensor of shape (..., n_freq, n_channels_1, n_channels_2)
        """
        return a.transpose(-3, -2) @ hermite(b.transpose(-3, -2))

    # for now, only supports determined case
    assert callable(model)

    # Pre-whitening

    # covariance matrix of input signal (n_freq, n_chan, n_chan)
    Cx = batch_abH(X, X) / n_frames
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
    Qinv_ref = Qinv[..., ref_mic, None, :]
    # Qinv_ref = e_vec[..., ref_mic, None, :] * e_val_sqrt[..., None, :]

    # keep the input for scaling
    X_in = X

    # The whitened input signal
    X = freq_wise_bmm(Q, X)

    # Pick the initial signal as the largest component of PCA,
    # i.e., the first channel (because we flipped the order above)
    Y = X[..., [0], :, :]

    Y = adjust_global_scale(Y, X_in[..., ref_mic, None, :, :])

    # projection back
    if ref_mic is not None:
        z = Q[..., 0, None, :] @ Cx[..., ref_mic, None]
        z = z.transpose(-3, -2)
        Y = z * Y

    for epoch in range(n_iter):

        if checkpoints_iter is not None and epoch in checkpoints_iter:
            checkpoints_list.append(X)

        # shape: (n_chan, n_freq, n_frames)
        # model takes as input a tensor of shape (..., n_frequencies, n_frames)
        weights = model(Y[..., 1:, :])

        # we normalize the sources to have source to have unit variance prior to
        # computing the model
        g = pt.clamp(pt.mean(mag_sq(Y), dim=(-2, -1), keepdim=True), min=1e-5)
        Y = divide(Y, pt.sqrt(g))
        weights = weights * g

        # compute the weighted spatial covariance matrix
        V = batch_abH(X[..., 1:, :] * weights, X[..., 1:, :]) / n_frames
        V = 0.5 * (V + hermite(V))

        # now compute the demixing vector
        # inv_eigenvalue, eigenvector = smallest_eigenvector_power_method(V, n_iter=5)
        inv_eigenvalue, eigenvector = smallest_eigenvector_eigh_cpu(V)
        inv_eval_sqrt = pt.sqrt(pt.clamp(inv_eigenvalue[..., None, None], min=1e-5))

        # eigenvector = eigenvector * inv_eval_sqrt  # dirty trick!

        # apply minimum distortion
        # we make use of the fact that eigenvector is length normalized
        z = hermite(eigenvector) @ Q[..., 1:, :, :] @ Cx[..., 1:, :, ref_mic, None]
        w = z * eigenvector * inv_eval_sqrt

        # dirty tricks ???
        # w = w * inv_eval_sqrt
        # w = w / (1.0 - 1.0 / inv_eigenvalue[..., None, None])  # Wiener filter

        """
        if ref_mic is not None and epoch == n_iter - 1:
            # scale based on projection back
            # (that seemed to work)
            # scale = Qinv_ref[..., 1:, :, :] @ eigenvector

            # scale based on projection back
            # (that I think should be correct)
            # scale = pt.conj(Qinv_ref[..., 1:, :, :] @ (eigenvector))

            # blind acoustic normalization (Warsitz & Haeb-Umback, 2007)
            # Vw = V @ eigenvector
            # num = pt.linalg.norm(Qinv[..., 1:, :, :] @ Vw, dim=(-2, -1)) * (
            # 1.0 / math.sqrt(w.shape[-2])
            # )
            # denom = pt.clamp(pt.abs(hermite(w) @ Vw), min=1e-7)
            # scale = num[..., None, None] / denom

            # apply the scale
            w = eigenvector * scale

        else:
            # no scale adjustment (updates as in ICASSP 2020 paper)
            w = eigenvector * inv_eval_sqrt
        """

        # the new output
        Y = freq_wise_bmm(hermite(w), X[..., 1:, :])

        """
        # Wiener filter to fix amplitude and phase
        num = pt.mean(
            Y * (1 - weights) * pt.conj(X_in[..., ref_mic, None, 1:, :]),
            dim=-1,
            keepdim=True,
        )
        denom = pt.clamp(pt.mean(Y.abs().square(), dim=-1, keepdim=True), min=1e-7)
        Y = Y * (num / denom)
        """

        # add the DC back
        pad_shape = Y.shape[:-2] + (1,) + Y.shape[-1:]
        Y = pt.cat((Y.new_zeros(pad_shape), Y), dim=-2)

        Y = adjust_global_scale(Y, X_in[..., ref_mic, None, :, :])

    return Y
