from typing import List, Optional

import torch

from .linalg import divide, eigh, hermite, inv_2x2, mag_sq
from .models import LaplaceModel
from .utils import select_most_energetic


def control_scale(X):
    # Here we control the scale of X
    g = torch.sqrt(1e-5 * torch.mean(mag_sq(X), dim=(-2, -1), keepdim=True))
    X = divide(X, g, eps=1e-5)
    X = torch.view_as_complex(
        torch.view_as_real(X) / torch.clamp(g[..., None], min=1e-5)
    )
    return X


def divide(num, denom, eps=1e-7):
    return torch.view_as_complex(
        torch.view_as_real(num) / torch.clamp(denom[..., None], min=eps)
    )


def spatial_model_update_iss(
    X: torch.Tensor,
    weights: torch.Tensor,
    W: Optional[torch.Tensor] = None,
    A: Optional[torch.Tensor] = None,
    eps: Optional[float] = 1e-3,
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
        v_num = (
            torch.einsum(
                "...cfn,...cfn,...fn->...cf", X, weights, X[..., s, :, :].conj()
            )
            / n_frames
        )
        v_denom = (
            torch.einsum("...cfn,...fn->...cf", weights, mag_sq(X[..., s, :, :]))
            / n_frames
        )

        v = v_num / torch.clamp(v_denom, min=eps)
        v_s = 1.0 - (1.0 / torch.sqrt(torch.clamp(v_denom[..., s, :], min=eps)))
        v = torch.cat((v[..., :s, :], v_s[..., None, :], v[..., s + 1 :, :]), dim=-2)

        # update demixed signals
        X = X - torch.einsum("...cf,...fn->...cfn", v, X[..., s, :, :])

        if W is not None:
            W = W - torch.einsum("...cf,...fd->...fcd", v, W[..., s, :])

        if A is not None:
            u = torch.einsum("...fcd,...df->...fc", A, v) / torch.clamp(
                1.0 - v_s[..., None], min=eps
            )
            A = torch.cat(
                (A[..., :s], A[..., [s]] + u[..., None], A[..., s + 1 :]), dim=-1
            )

    return X, W, A


def spatial_model_update_ip2(
    Xo: torch.Tensor,
    weights: torch.Tensor,
    W: Optional[torch.Tensor] = None,
    A: Optional[torch.Tensor] = None,
    eps: Optional[float] = 1e-7,
):
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
        Vloc = torch.einsum(
            "...fn,...cfn,...dfn->...fcd", weights[..., k, :, :], Xo, Xo.conj()
        )
        Vloc = Vloc / Xo.shape[-1]
        # make sure V is hermitian symmetric
        Vloc = 0.5 * (Vloc + hermite(Vloc))
        V.append(Vloc)

    eigval, eigvec = eigh(V[1], V[0], eps=eps)

    # reverse order of eigenvectors
    eigvec = torch.flip(eigvec, dims=(-1,))

    scale_0 = abs(
        torch.conj(eigvec[..., None, :, 0]) @ (V[0] @ eigvec[..., :, None, 0])
    )
    scale_1 = abs(
        torch.conj(eigvec[..., None, :, 1]) @ (V[1] @ eigvec[..., :, None, 1])
    )
    scale = torch.cat((scale_0, scale_1), dim=-1)
    scale = torch.clamp(torch.sqrt(torch.clamp(scale, min=1e-7)), min=eps)
    eigvec = eigvec / scale

    if W is not None:
        W = hermite(eigvec)
        if A is not None:
            A = inv_2x2(W)

    X = torch.einsum("...fcd,...dfn->...cfn", hermite(eigvec), Xo)

    return X, W, A


def auxiva_iss(
    X: torch.Tensor,
    n_iter: Optional[int] = 20,
    n_src: Optional[int] = None,
    model: Optional[callable] = None,
    eps: Optional[float] = 1e-6,
    two_chan_ip2: Optional[bool] = True,
    proj_back_mic: Optional[bool] = 0,
    checkpoints_iter: Optional[List[int]] = None,
    checkpoints_list: Optional[List] = None,
) -> torch.Tensor:

    """
    Blind source separation based on independent vector analysis with
    alternating updates of the mixing vectors

    Parameters
    ----------
    X: Tensor, shape (..., n_channels, n_frequencies, n_frames)
        STFT representation of the signal
    n_iter: int, optional
        The number of iterations (default 20)
    n_src: int, optional
        When n_src is less than the number of channels, the n_src most energetic
        are selected for the output
    model: SourceModel
        The model of source distribution (default: Laplace)
    eps: float
        A small constant to make divisions and the like numerically stable
    two_chan_ip2: bool
        For the 2 channel case, use the more efficient IP2 algorithm (default: True).
        Ignored when using more than 2 channels.
    proj_back_mic: int
        The microphone index to use as a reference when adjusting the scale and delay
    checkpoints_iter: List of int
        Optionally, we can keep intermediate results for later analysis
        Should be used together with checkpoints_list
    checkpoints_list: List
        An empty list can be passed and the intermediate signals are
        appended to the list for iterations numbers contained in checkpoints_iter

    Returns
    -------
    X: Tensor, shape (..., n_channels, n_frequencies, n_frames)
        STFT representation of the signal after separation
    """

    n_chan, n_freq, n_frames = X.shape[-3:]

    if model is None:
        model = LaplaceModel()

    # for now, only supports determined case
    assert callable(model)

    if n_chan == 2 and two_chan_ip2:
        Xo = X

    if proj_back_mic is not None:
        assert (
            0 <= proj_back_mic < n_chan
        ), "The reference microphone index must be between 0 and # channels - 1."
        W = X.new_zeros(X.shape[:-3] + (n_freq, n_chan, n_chan))
        W[:] = torch.eye(n_chan).type_as(W)
        A = W.clone()
    else:
        W = None
        A = None

    for epoch in range(n_iter):

        if checkpoints_iter is not None and epoch in checkpoints_iter:
            checkpoints_list.append(X)

        # shape: (n_chan, n_freq, n_frames)
        # model takes as input a tensor of shape (..., n_frequencies, n_frames)
        weights = model(X)

        # we normalize the sources to have source to have unit variance prior to
        # computing the model
        g = torch.clamp(torch.mean(mag_sq(X), dim=(-2, -1), keepdim=True), min=1e-5)
        X = divide(X, torch.sqrt(g))
        weights = weights * g

        if n_chan == 2 and two_chan_ip2:
            # Here are the exact/fast updates for two channels using the GEVD
            X, W, A = spatial_model_update_ip2(Xo, weights, W=W, A=A, eps=eps)

        else:
            # Iterative Source Steering updates
            X, W, A = spatial_model_update_iss(X, weights, W=W, A=A, eps=eps)

    if proj_back_mic is not None:
        a = A[..., :, [proj_back_mic], :].moveaxis(-1, -3)
        X = a * X

    if n_src is not None and n_src < n_chan:
        # select sources based on energy
        X = select_most_energetic(X, num=n_src, dim=-3, dim_reduc=(-2, -1))

    return X
