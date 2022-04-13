from typing import Optional
import torch
from .linalg import eigh, solve_loaded, bmm, hermite, multiply


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
    eps: Optional[float] = 1e-5,
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


def compute_gev_bf(X, mask, ref_mic=0):
    """
    X: (batch, channels, freq, frames)
    mask: (batch, sources, freq, frames)
    """

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

    # now compute the GEV beamformers
    # eigenvalues are listed in ascending order
    eigval, eigvec = eigh(R_target, R_noise)

    # compute the scale
    rhs = torch.eye(eigvec.shape[-1], dtype=eigvec.dtype, device=eigvec.device)
    rhs = rhs[:, -1, None]
    steering_vector = torch.linalg.solve(hermite(eigvec), rhs)
    scale = steering_vector[..., ref_mic, None, :]

    # (batch, freq, sources, channels)
    gev_bf = torch.conj(eigvec[..., -1].transpose(-3, -2)) * scale

    return gev_bf


class MVDRBeamformer(nn.Module):
    """
    Implementation of MVDR beamformer described in

    C. Boeddeker et al., "CONVOLUTIVE TRANSFER FUNCTION INVARIANT SDR TRAINING
    CRITERIA FOR MULTI-CHANNEL REVERBERANT SPEECH SEPARATION", Proc. ICASSP
    2021.

    Parameters
    ----------
    n_fft: int
        FFT size for the STFT
    hop_length: int
        Shift of the STFT
    window: str
        The window to use for the STFT
    mask_model: torch.nn.Module
        A function that is given one spectrogram and returns 3 masks
        of the same size as the input
    ref_mic: int, optionala
        Reference channel (default: 0)
    use_n_power_iter: int, optional
        Use the power iteration method to compute the relative
        transfer function instead of the full generalized
        eigenvalue decomposition (GEVD). The number of iteration
        desired should be provided. By default, the full GEVD
        is used
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: Optional[int] = None,
        window: Optional[int] = None,
        mask_model: Optional[Union[torch.nn.Module, Dict]] = None,
        ref_mic: Optional[int] = 0,
        use_n_power_iter: Optional[int] = None,
    ):
        super().__init__()
        # other attributes
        self.ref_mic = ref_mic
        self.n_power_iter = use_n_power_iter

        if isinstance(mask_model, dict):
            self.mask_model = module_from_config(**mask_model)
        elif isinstance(mask_model, nn.Module):
            self.mask_model = mask_model

        # the stft engine
        self.stft = STFT(n_fft, hop_length=hop_length, window=window)


    def forward(self, x):

        X = self.stft(x)

        # remove the DC
        X = X[..., 1:, :]

        # compute the masks (..., n_src, n_masks, n_freq, n_frames)
        masks = self.mask_model(X[..., self.ref_mic, :, :])
        masks = [masks[..., i, :, :] for i in range(3)]

        # compute the covariance matrices
        R_tgt, R_noise_1, R_noise_2 = [
            torch.einsum("...sfn,...cfn,...dfn->...sfcd", mask, X, X.conj())
            for mask in masks
        ]

        # compute the relative transfer function
        rtf = compute_mvdr_rtf_eigh(
            R_tgt, R_noise_2, power_iterations=self.n_power_iter
        )

        # compute the beamforming weights
        # shape (..., n_freq, n_chan)
        bf = compute_mvdr_bf(R_noise_1, rtf)

        # compute output
        X = torch.einsum("...cfn,...sfc->...sfn", X, bf.conj())

        # add back DC offset
        pad_shape = X.shape[:-2] + (1,) + X.shape[-1:]
        X = torch.cat((X.new_zeros(pad_shape), X), dim=-2)

        y = self.stft.inv(X)

        if y.shape[-1] < x.shape[-1]:
            y = torch.cat(
                (y, y.new_zeros(y.shape[:-1] + (x.shape[-1] - y.shape[-1],))), dim=-1
            )
        elif y.shape[-1] > x.shape[-1]:
            y = y[..., : x.shape[-1]]

        return y


class MWFBeamformer(nn.Module):
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
        n_fft: int,
        hop_length: Optional[int] = None,
        window: Optional[int] = None,
        mask_model: Optional[Union[torch.nn.Module, Dict]] = None,
        ref_mic: Optional[int] = 0,
        time_invariant: Optional[bool] = True,
    ):
        super().__init__()
        # other attributes
        self.ref_mic = ref_mic
        self.time_invariant = time_invariant

        if isinstance(mask_model, dict):
            self.mask_model = module_from_config(**mask_model)
        elif isinstance(mask_model, nn.Module):
            self.mask_model = mask_model

        # the stft engine
        self.stft = STFT(n_fft, hop_length=hop_length, window=window)

    def forward(self, x):

        X = self.stft(x)

        # remove the DC
        X = X[..., 1:, :]

        # compute the masks (..., n_src, n_masks, n_freq, n_frames)
        masks = self.mask_model(X[..., self.ref_mic, :, :])

        covmat_masks = [masks[..., i, :, :] for i in range(2)]

        # compute the covariance matrices
        R_tgt, R_noise = [
            torch.einsum("...sfn,...cfn,...dfn->...sfcd", mask, X, X.conj())
            for mask in covmat_masks
        ]

        if self.time_invariant:

            # compute the beamforming weights
            # shape (..., n_freq, n_chan)
            bf = compute_mwf_bf(R_tgt, R_noise, ref_mic=self.ref_mic)

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
            bf = compute_mwf_bf(R_tgt, R_noise, ref_mic=self.ref_mic)

            # compute output
            X = torch.einsum("...cfn,...sfnc->...sfn", X, bf.conj())

        # add back DC offset
        pad_shape = X.shape[:-2] + (1,) + X.shape[-1:]
        X = torch.cat((X.new_zeros(pad_shape), X), dim=-2)

        y = self.stft.inv(X)

        if y.shape[-1] < x.shape[-1]:
            y = torch.cat(
                (y, y.new_zeros(y.shape[:-1] + (x.shape[-1] - y.shape[-1],))), dim=-1
            )
        elif y.shape[-1] > x.shape[-1]:
            y = y[..., : x.shape[-1]]

        return y



class MaskGEVBeamformer(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: Optional[int] = None,
        window: Optional[int] = None,
        mask_model: Optional[torch.nn.Module] = None,
        ref_mic: Optional[int] = 0,
        mdp_p: Optional[float] = None,
        mdp_q: Optional[float] = None,
        proj_back: Optional[bool] = False,
        mdp_phase: Optional[bool] = False,
        mdp_model: Optional[bool] = None,
    ):

        super().__init__()

        self.mask_model = mask_model

        # other attributes
        self.ref_mic = ref_mic
        self.mdp_p = mdp_p
        self.mdp_q = mdp_q
        self.proj_back = proj_back
        self.mdp_phase = mdp_phase
        self.mdp_model = mdp_model

        # the stft engine
        self.stft = STFT(n_fft, hop_length=hop_length, window=window)


    def forward(self, x):

        # STFT (batch, channels, freq, frames)
        X = self.stft(x)

        # Compute the mask
        mask = self.mask_model(X[..., self.ref_mic, :, :])

        # Compute the beamformers
        gev_bf = compute_gev_bf(X, mask, ref_mic=self.ref_mic)

        # Separation
        # (batch, sources, freq, frames)
        Y = bmm(gev_bf, X.transpose(-3, -2)).transpose(-3, -2)

        # Restore the scale
        if self.proj_back:
            Y = projection_back(Y, X[..., self.ref_mic, :, :])
        elif self.mdp_phase:
            Y = minimum_distortion_l2_phase(Y, X[..., self.ref_mic, :, :],)
        elif self.mdp_p is not None:
            Y = minimum_distortion(
                Y,
                X[..., self.ref_mic, :, :],
                p=self.mdp_p,
                q=self.mdp_q,
                model=self.mdp_model,
                max_iter=10,
            )

        # iSTFT
        y = self.stft.inv(Y)  # (n_samples, n_channels)

        return y
