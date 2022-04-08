from typing import Optional
import torch
from .linalg import eigh, solve_loaded


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



def compute_mvdr_rtf_eigh(
    covmat_target: torch.Tensor,
    covmat_noise: torch.Tensor,
    ref_mic: Optional[int] = 0,
    power_iterations: Optional[int] = None,
) -> torch.Tensor:

    """
    Compute the Relative Transfer Function

    Parameters
    ----------
    covmat_target: torch.Tensor, (..., n_channels, n_channels)
        The covariance matrices of the target signal
    covmat_noise: torch.Tensor, (..., n_channels, n_channels)
        The covariance matrices of the noise signal
    ref_mic: int
        The channel used as the reference
    power_iterations: int, optional
        An integer can be provided. If it is provided, a power iteration
        algorithm is used instead of the generalized eigenvalue decomposition (GEVD).
        If it is not provided, the regular algorithm for GEVD is used.
    """

    if power_iterations is None:
        eigval, eigvec = eigh(covmat_target, covmat_noise)
        steering_vector = eigvec[..., :, -1]
    else:
        Phi = solve_loaded(covmat_target, covmat_noise, load=1e-5)

        # use power iteration to compute dominant eigenvector
        v = Phi[..., :, 0]
        for epoch in range(power_iterations - 1):
            v = torch.einsum("...cd,...d->...c", Phi, v)

        steering_vector = torch.einsum("...cd,...d->...c", covmat_noise, v)

    # normalize reference component
    scale = steering_vector[..., [ref_mic]]
    steering_vector = steering_vector / scale

    return steering_vector


def compute_mvdr_bf(
    covmat_noise: torch.Tensor,
    steering_vector: torch.Tensor,
    eps: Optional[float] = 1e-6,
) -> torch.Tensor:
    """
    Computes the beamforming weights of the Minimum Variance Distortionles Response (MVDR) beamformer

    Parameters
    ----------
    """
    a = solve_loaded(covmat_noise, steering_vector, load=1e-5)
    denom = torch.einsum("...c,...c->...", steering_vector.conj(), a)
    denom = denom.real  # because covmat is PSD
    denom = torch.clamp(denom, min=1e-6)
    return a / denom[..., None]