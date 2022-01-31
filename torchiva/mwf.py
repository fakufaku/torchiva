from typing import Optional
import torch

from .linalg import solve_loaded


def compute_mwf_bf(
    covmat_target: torch.Tensor,
    covmat_noise: torch.Tensor,
    ref_mic: Optional[int] = None,
):
    """
    Compute the multichannel Wiener filter (MWF) for a given target
    covariance matrix and noise covariance matrix.

    Parameters
    ----------
    covmat_target: torch.Tensor, (..., n_channels, n_channels)
        The covariance matrices of the target signal
    covmat_noise: torch.Tensor, (..., n_channels, n_channels)
        The covariance matrices of the noise signal
    ref_mic: int, optional
        If a reference channel is provide, only the filter corresponding to
        that channel is computed.  If it is not provided, all the filters are
        computed and the returned tensor is a square matrix.

    Returns
    -------
    mwf: torch.Tensor, (..., n_channels, n_channels) or (..., n_channels)
        The multichannel Wiener filter, if ref_mic is not provide, the last two
        dimensions for a square matrix the size of the number of channels.  If
        ref_mic is provide, then there is one less dimension, and the length of
        the last dimension the number of channels.
    """

    if ref_mic is None:
        W_H = solve_loaded(covmat_target + covmat_noise, covmat_target, load=1e-5)
        return W_H

    else:
        w = solve_loaded(covmat_target + covmat_noise, covmat_target[..., ref_mic])
        return w
