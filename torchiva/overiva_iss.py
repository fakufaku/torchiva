from typing import List, Optional

import torch as pt

from .base import SourceModelBase
from .linalg import bmm, divide, eigh, hermite, mag, mag_sq, multiply, solve_loaded
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


def orthogonal_constraint(W_top, Cx, load=1e-4):
    n_src, n_chan = W_top.shape[-2:]

    if n_src == n_chan:
        return W_top
    elif n_chan < n_src:
        raise ValueError(
            "OverIVA requires the number of sources to be "
            "less or equal than that of microphones"
        )

    # create a new demixing matrix
    W = W_top.new_zeros(Cx.shape)
    W[..., :n_src, :] = W_top
    W[..., n_src:, n_src:] = -pt.eye(n_chan - n_src)

    # compute the missing part
    tmp = W_top @ Cx
    W[..., n_src:, :n_src] = hermite(
        solve_loaded(tmp[..., :n_src], tmp[..., n_src:], load=load)
    )

    return W


def projection_back_from_demixing_matrix(
    Y: pt.Tensor, W: pt.Tensor, ref_mic: Optional[int] = 0, load: Optional[float] = 1e-4
) -> pt.Tensor:
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


    if n_src == n_chan:
        eye = pt.eye(n_chan, n_chan).type_as(W)
        e1 = eye[..., :, [ref_mic]]
        a = solve_loaded(W.transpose(-2, -1), e1)  # (..., n_freq, n_chan, 1)
        a = a.transpose(-3, -2)

    else:
        A = W[..., :n_src, :n_src]
        B = W[..., :n_src, n_src:]
        C = W[..., n_src:, :n_src]

        if ref_mic < n_src:
            eye = pt.eye(n_src, n_src).type_as(W)
            e1 = eye[:, [ref_mic]]
        else:
            e1 = C[..., [ref_mic - n_src], :].transpose(-2, -1)

        WW = A + B @ C
        a = solve_loaded(WW.transpose(-2, -1), e1)
        a = a.transpose(-3, -2)

    Y = Y * a

    return Y, a


def overiva(
    X: pt.Tensor,
    n_iter: Optional[int] = 20,
    n_src: Optional[int] = None,
    model: Optional[SourceModelBase] = None,
    eps: Optional[float] = None,
    ref_mic: Optional[int] = 0,
    proj_back: Optional[bool] = False,
    use_wiener: Optional[bool] = False,
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

    # remove DC part
    X = X[..., 1:, :]

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
    Y = X[..., :n_src, :, :]  # sign to be consistant with W

    # covariance matrix of input signal (n_freq, n_chan, n_chan)
    Cx = pt.einsum("...cfn,...dfn->...fcd", X, X.conj()) / n_frames

    # apply the orthogonal constraint
    W = orthogonal_constraint(W_top, Cx, load=1e-4)

    for epoch in range(n_iter):

        if checkpoints_iter is not None and epoch in checkpoints_iter:
            checkpoints_list.append(X)

        if use_wiener:
            wiener_weights = []


        # shape: (n_chan, n_freq, n_frames)
        # model takes as input a tensor of shape (..., n_frequencies, n_frames)
        weights = model(Y)

        # we normalize the sources to have source to have unit variance prior to
        # computing the model
        """
        g = pt.clamp(pt.mean(mag_sq(Y), dim=(-2, -1), keepdim=True), min=1e-5)
        Y = divide(Y, pt.sqrt(g))
        weights = weights * g
        """

        for k in range(n_src):

            # compute the weighted spatial covariance matrix
            V = batch_abH(X * weights[..., [k], :, :], X)

            # solve for the new demixing vector
            WV = W @ V

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

            # re-build the demixing matrix
            W_top = pt.cat([W[..., :k, :], new_w, W[..., k + 1 : n_src, :]], dim=-2)

            # apply the orthogonal constraint
            W = orthogonal_constraint(W_top, Cx, load=1e-4)

            if use_wiener:
                new_w = new_w[..., 0, :]
                output_pwr = pt.einsum("...fc,...fcd,...fd->...f", new_w, Cx, new_w.conj()).real
                noise_pwr = pt.einsum("...fc,...fcd,...fd->...f", new_w, V, new_w.conj()).real
                weight = 1 - noise_pwr / pt.clamp(output_pwr, min=1e-5)
                wiener_weights.append(weight[..., None, :])

        # demix
        Y = freq_wise_bmm(W_top, X)

        if proj_back:
            Y, a = projection_back_from_demixing_matrix(
                Y, W, ref_mic=ref_mic, load=1e-4
            )

        if use_wiener:
            wiener_weights = pt.cat(wiener_weights, dim=-2)
            Y = Y * wiener_weights[..., None]

    # add back DC offset
    pad_shape = Y.shape[:-2] + (1,) + Y.shape[-1:]
    Y = pt.cat((Y.new_zeros(pad_shape), Y), dim=-2)

    return Y


def auxiva_ip(
    X: pt.Tensor,
    n_iter: Optional[int] = 20,
    model: Optional[SourceModelBase] = None,
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
