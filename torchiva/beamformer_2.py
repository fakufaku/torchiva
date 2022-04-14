from typing import Optional
import torch
from .linalg import eigh, solve_loaded, bmm, hermite, multiply
from .base import BFBase


def compute_mwf_bf(
    covmat_target: torch.Tensor,
    covmat_noise: torch.Tensor,
    eps: Optional[float] = 1e-5,
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
        W_H = solve_loaded(covmat_target + covmat_noise, covmat_target, load=eps)
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
        v = eigvec[..., :, -1]
        #steering_vector = eigvec[..., :, -1]
        
    else:
        #Phi = solve_loaded(covmat_target, covmat_noise, load=1e-5) #original
        Phi = solve_loaded(covmat_noise, covmat_target, load=1e-5)

        # use power iteration to compute dominant eigenvector
        v = Phi[..., :, 0]
        for epoch in range(power_iterations - 1):
            v = torch.einsum("...cd,...d->...c", Phi, v)
            v = torch.nn.functional.normalize(v, dim=-1)
            
    steering_vector = torch.einsum("...cd,...d->...c", covmat_noise, v)
    #steering_vector = torch.einsum("...cd,...d->...c", covmat_target, v)
    #steering_vector = v

    # normalize reference component
    scale = steering_vector[..., [ref_mic]]
    steering_vector = steering_vector / scale

    return steering_vector


def compute_mvdr_bf(
    covmat_noise: torch.Tensor,
    steering_vector: torch.Tensor,
    eps: Optional[float] = 1e-5,
) -> torch.Tensor:
    """
    Computes the beamforming weights of the Minimum Variance Distortionles Response (MVDR) beamformer

    Parameters
    ----------
    """
    a = solve_loaded(covmat_noise, steering_vector, load=eps)
    denom = torch.einsum("...c,...c->...", steering_vector.conj(), a)
    denom = denom.real  # because covmat is PSD
    denom = torch.clamp(denom, min=eps)
    return a / denom[..., None]


def compute_mvdr_bf2(
    covmat_target: torch.Tensor,
    covmat_noise: torch.Tensor,
    ref_mic: Optional[int] = 0,
    eps: Optional[float] = 1e-5,

) -> torch.Tensor:

    num = torch.linalg.solve(covmat_noise, covmat_target)
    denom = torch.sum(torch.diagonal(num,dim1=-2,dim2=-1),dim=-1)

    w = num / denom[...,None,None]

    return w[..., ref_mic]


def compute_gev_bf(
    covmat_target: torch.Tensor,
    covmat_noise: torch.Tensor,
    ref_mic: Optional[int] = 0,
):

    # now compute the GEV beamformers
    # eigenvalues are listed in ascending order
    eigval, eigvec = eigh(covmat_target, covmat_noise)

    # compute the scale
    rhs = torch.eye(eigvec.shape[-1], dtype=eigvec.dtype, device=eigvec.device)
    rhs = rhs[:, -1, None]
    steering_vector = torch.linalg.solve(hermite(eigvec), rhs)
    scale = steering_vector[..., ref_mic, :].transpose(-3, -2)

    # (batch, freq, sources, channels)
    gev_bf = torch.conj(eigvec[..., -1].transpose(-3, -2)) * scale

    return gev_bf



class MVDRBeamformer(BFBase):
    """
    Implementation of MVDR beamformer described in

    C. Boeddeker et al., "CONVOLUTIVE TRANSFER FUNCTION INVARIANT SDR TRAINING
    CRITERIA FOR MULTI-CHANNEL REVERBERANT SPEECH SEPARATION", Proc. ICASSP
    2021.

    Parameters
    ----------
    mask_model: torch.nn.Module
        A function that is given one spectrogram and returns 3 masks
        of the same size as the input
    ref_mic: int, optional
        Reference channel (default: ``0``)
    n_power_iter: int, optional
        Use the power iteration method to compute the relative
        transfer function instead of the full generalized
        eigenvalue decomposition (GEVD). The number of iteration
        desired should be provided. If set to ``None``, the full GEVD
        is used (default: ``None``).
    """

    def __init__(
        self,
        mask_model: torch.nn.Module,
        ref_mic: Optional[int] = 0,
        eps: Optional[float] = 1e-5,
        n_power_iter: Optional[int] = None,
    ):
        super().__init__(
            mask_model,
            ref_mic=ref_mic,
            eps=eps,
        )

        self.n_power_iter=n_power_iter


    def forward(
        self, 
        X: torch.Tensor,
        mask_model: Optional[torch.nn.Module] = None,
        ref_mic: Optional[int] = None,
        eps: Optional[float] = None,
        n_power_iter: Optional[int] = None,
    ):

        mask_model, ref_mic, eps, n_power_iter = self._set_params(
            mask_model, ref_mic, eps, n_power_iter,
        )

        # remove the DC
        X = X[..., 1:, :]

        # compute the masks (..., n_src, n_masks, n_freq, n_frames)
        masks = mask_model(X[..., ref_mic, :, :])
        masks = [masks[..., i, :, :] for i in range(3)]

        # compute the covariance matrices
        R_tgt, R_noise_1, R_noise_2 = [
            torch.einsum("...sfn,...cfn,...dfn->...sfcd", mask, X, X.conj())
            for mask in masks
        ]

        # compute the relative transfer function
        rtf = compute_mvdr_rtf_eigh(
            R_tgt, R_noise_2, ref_mic=ref_mic, power_iterations=n_power_iter,
        )

        # compute the beamforming weights
        # shape (..., n_freq, n_chan)
        bf = compute_mvdr_bf(R_noise_1, rtf, eps=eps)

        # compute output
        X = torch.einsum("...cfn,...sfc->...sfn", X, bf.conj())

        # add back DC offset
        pad_shape = X.shape[:-2] + (1,) + X.shape[-1:]
        X = torch.cat((X.new_zeros(pad_shape), X), dim=-2)

        return X


class MWFBeamformer(BFBase):
    """
    Implementation of the MWF beamformer described in

    Y. Masuyama et al., "CONSISTENCY-AWARE MULTI-CHANNEL SPEECH ENHANCEMENT
    USING DEEP NEURAL NETWORKS", Proc. ICASSP 2020.

    Parameters
    ----------
    n_fft: int
        FFT size for the STFT
    hop_length: int
        Shift of the STFT
    window: str
        The window to use for the STFT
    mask_model: torch.nn.Module
        A function that is given one spectrogram and returns 2 masks
        of the same size as the input, (or 4 for the time-variant version)
    ref_mic: int, optionala
        Reference channel (default: 0)
    time_invariant: bool, optional
        If set to true, this flag indicates that we want to use the
        time-invariant version of MWF (default).  If set to false, the
        time-varying MWF is used instead
    """

    def __init__(
        self,
        mask_model: torch.nn.Module,
        ref_mic: Optional[int] = 0,
        eps: Optional[float] = 1e-5,
        time_invariant: Optional[bool] = True,
    ):
        super().__init__(
            mask_model,
            ref_mic=ref_mic,
            eps=eps,
        )

        self.time_invariant = time_invariant


    def forward(
        self, 
        X: torch.Tensor,
        mask_model: Optional[torch.nn.Module] = None,
        ref_mic: Optional[int] = None,
        eps: Optional[float] = None,
        time_invariant: Optional[bool] = None,
    ):

        mask_model, ref_mic, eps, time_invariant = self._set_params(
            mask_model, ref_mic, eps, time_invariant,
        )

        # remove the DC
        X = X[..., 1:, :]

        # compute the masks (..., n_src, n_masks, n_freq, n_frames)
        masks = mask_model(X[..., ref_mic, :, :])

        covmat_masks = [masks[..., i, :, :] for i in range(2)]

        # compute the covariance matrices
        R_tgt, R_noise = [
            torch.einsum("...sfn,...cfn,...dfn->...sfcd", mask, X, X.conj())
            for mask in covmat_masks
        ]

        if time_invariant:

            # compute the beamforming weights
            # shape (..., n_freq, n_chan)
            bf = compute_mwf_bf(R_tgt, R_noise, eps=eps, ref_mic=ref_mic)

            # compute output
            X = torch.einsum("...cfn,...sfc->...sfn", X, bf.conj())

        else:
            assert masks.shape[-3] == 4
            V_tgt, V_noise = [masks[..., i, :, :] for i in range(2, 4)]

            # new shape is (..., n_src, n_freq, n_frames, n_chan, n_chan)
            R_tgt = R_tgt[..., :, None, :, :] * V_tgt[..., :, :, None, None]
            R_noise = R_noise[..., :, None, :, :] * V_noise[..., :, :, None, None]

            # compute the beamforming weights
            # shape (..., n_freq, n_time, n_chan)
            bf = compute_mwf_bf(R_tgt, R_noise, eps=eps, ref_mic=ref_mic)

            # compute output
            X = torch.einsum("...cfn,...sfnc->...sfn", X, bf.conj())

        # add back DC offset
        pad_shape = X.shape[:-2] + (1,) + X.shape[-1:]
        X = torch.cat((X.new_zeros(pad_shape), X), dim=-2)

        return X



class GEVBeamformer(BFBase):
    def __init__(
        self,
        mask_model: torch.nn.Module,
        ref_mic: Optional[int] = 0,
        eps: Optional[float] = 1e-5,
    ):

        super().__init__(
            mask_model,
            ref_mic=ref_mic,
            eps=eps,
        )

    def forward(
        self, 
        X: torch.Tensor,
        mask_model: Optional[torch.nn.Module] = None,
        ref_mic: Optional[int] = None,
        eps: Optional[float] = None,
    ):

        mask_model, ref_mic, eps = self._set_params(
            mask_model, ref_mic, eps,
        )

        # Compute the mask
        mask = mask_model(X[..., ref_mic, :, :])

        # (batch, sources, channels, freq, frames)
        targets = multiply(X[..., None, :, :, :], mask[..., :, None, :, :])
        noise = multiply(X[..., None, :, :, :], (1.0 - mask[..., :, None, :, :]))

        # (batch, sources, channels, freq, frames)
        targets = targets.transpose(-3, -2)
        noise = noise.transpose(-3, -2)

        # create the covariance matrices
        # (batch, sources, freq, channels, channels)
        R_target = bmm(targets, hermite(targets)) / targets.shape[-1]
        R_noise = bmm(noise, hermite(noise)) / targets.shape[-1]

        # Compute the beamformers
        gev_bf = compute_gev_bf(R_target, R_noise, ref_mic=ref_mic)

        # Separation
        # (batch, sources, freq, frames)
        Y = bmm(gev_bf, X.transpose(-3, -2)).transpose(-3, -2)

        return Y
