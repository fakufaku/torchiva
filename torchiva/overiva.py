from typing import List, Optional, Callable

import torch as pt

from .linalg import bmm, divide, eigh, hermite, mag, mag_sq, multiply
from .models import LaplaceModel
from .parameters import eps_models


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
    ----------
    a: (..., n_channels_1, n_freq, n_frames)
        op 1
    b: (..., n_channels_2, n_freq, n_frames)
        op 2

    Returns
    -------
    Tensor of shape (..., n_freq, n_channels_1, n_channels_2)
    """
    return a.transpose(-3, -2) @ hermite(b.transpose(-3, -2))


def orthogonal_constraint(W_top, Cx):
    n_src, n_chan = W_top.shape[-2:]

    # create a new demixing matrix
    W = W_top.new_zeros(Cx.shape)
    W[..., :n_src, :] = W_top
    W[..., n_src:, n_src:] = -pt.eye(n_chan - n_src)

    # compute the missing part
    tmp = W_top @ Cx
    dload = 1e-7 * pt.eye(n_src, dtype=tmp.dtype, device=tmp.device)
    W[..., n_src:, :n_src] = hermite(
        pt.linalg.solve(tmp[..., :n_src] + dload, tmp[..., n_src:])
    )

    return W


def overiva(
    X: pt.Tensor,
    n_iter: Optional[int] = 20,
    n_src: Optional[int] = None,
    model: Optional[Callable] = None,
    eps: Optional[float] = None,
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

    batch_shape = X.shape[:-3]
    n_chan, n_freq, n_frames = X.shape[-3:]

    if n_src is None:
        n_src = n_chan

    if eps is None:
        eps = eps_models["laplace"]

    if model is None:
        model = LaplaceModel()

    # for now, only supports determined case
    assert callable(model)

    W_top = X.new_zeros(batch_shape + (n_freq, n_src, n_chan))
    # minus sign so that the parametrization is correct for overiva
    W_top[:] = pt.eye(n_src, n_chan)

    # initial estimate
    Y = X[..., :n_src, :, :]  # sign to be consitant with W

    # covariance matrix of input signal (n_freq, n_chan, n_chan)
    Cx = batch_abH(X, X) / n_frames

    # apply the orthogonal constraint
    if n_src < n_chan:
        W = orthogonal_constraint(W_top, Cx)
    else:
        W = W_top

    for epoch in range(n_iter):

        if checkpoints_iter is not None and epoch in checkpoints_iter:
            checkpoints_list.append(X)

        # shape: (n_chan, n_freq, n_frames)
        # model takes as input a tensor of shape (..., n_frequencies, n_frames)
        weights = model(Y)

        # we normalize the sources to have source to have unit variance prior to
        # computing the model
        g = pt.clamp(pt.mean(mag_sq(Y), dim=(-2, -1), keepdim=True), min=1e-5)
        Y = divide(Y, pt.sqrt(g))
        weights = weights * g

        for k in range(n_src):

            # compute the weighted spatial covariance matrix
            V = batch_abH(X * weights[..., [k], :, :], X) / n_frames

            # remove DC
            V = V[..., 1:, :, :]

            # solve for the new demixing vector
            WV = W[..., 1:, :, :] @ V

            # the new filter, unscaled
            new_w = pt.conj(
                pt.linalg.solve(
                    WV + 1e-4 * pt.eye(n_chan, dtype=W.dtype, device=W.device),
                    pt.eye(n_chan, dtype=W.dtype, device=W.device)[:, k],
                )
            )

            # resolve scale
            scale = pt.abs(new_w[..., None, :] @ V @ hermite(new_w[..., None, :]))
            new_w = new_w[..., None, :] / pt.sqrt(pt.clamp(scale, min=1e-5))

            # add the DC back
            pad_shape = new_w.shape[:-3] + (1,) + new_w.shape[-2:]
            new_w = pt.cat([new_w.new_zeros(pad_shape), new_w], dim=-3)

            # re-build the demixing matrix
            W_top = pt.cat([W[..., :k, :], new_w, W[..., k + 1 : n_src, :]], dim=-2)

            # apply the orthogonal constraint
            if n_src < n_chan:
                W = orthogonal_constraint(W_top, Cx)
            else:
                W = W_top

        # demix
        Y = freq_wise_bmm(W_top, X)

    return Y


def auxiva_ip(
    X: pt.Tensor,
    n_iter: Optional[int] = 20,
    model: Optional[Callable] = None,
    eps: Optional[float] = None,
    checkpoints_iter: Optional[List[int]] = None,
    checkpoints_list: Optional[List] = None,
) -> pt.Tensor:
    return overiva(
        X,
        n_iter=n_iter,
        model=model,
        eps=eps,
        checkpoints_iter=checkpoints_iter,
        checkpoints_list=checkpoints_list,
    )
