# Copyright (c) 2022 Robin Scheibler, Kohei Saijo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

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
        # use GEVD to obtain eigenvectors
        eigval, eigvec = eigh(covmat_target, covmat_noise)
        v = eigvec[..., :, -1]

    else:
        # use power iteration to compute dominant eigenvector
        Phi = solve_loaded(covmat_noise, covmat_target, load=1e-5)
        v = Phi[..., :, 0]
        for epoch in range(power_iterations - 1):
            v = torch.einsum("...cd,...d->...c", Phi, v)
            v = torch.nn.functional.normalize(v, dim=-1)

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
    denom = torch.sum(torch.diagonal(num, dim1=-2, dim2=-1), dim=-1)
    w = num / denom[..., None, None]

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
    Implementation of MVDR beamformer.
    This class is basically assumes DNN-based beamforming.
    also supports the case of estimating three masks

    Parameters
    ----------
    mask_model: torch.nn.Module
        A function that is given one spectrogram and returns 2 or 3 masks
        of the same size as the input.
        When 3 masks (1 for target and the rest 2 for noise) are etimated,
        they are utilized as in [10]_
    ref_mic: int, optional
        Reference channel (default: ``0``)
    eps: float, optional
        A small constant to make divisions
        and the like numerically stable (default:``1e-5``).
    mvdr_type: str, optional
        The way to obtain the MVDR weight.
        If set to ``rtf``, relative transfer function is computed
        to obtain MVDR. If set to 'scm', MVDR weight is obtained
        directly with spatial covariance matrices [11]_ (default: ``rtf``).
    n_power_iter: int, optional
        Use the power iteration method to compute the relative
        transfer function instead of the full generalized
        eigenvalue decomposition (GEVD). The number of iteration
        desired should be provided. If set to ``None``, the full GEVD
        is used (default: ``None``).

    Methods
    --------
    forward(X, mask_model = None, ref_mic = None, eps = None, mvdr_type = None, n_power_iter = None)

    Parameters
    ----------
    X: torch.Tensor
        The input mixture in STFT-domain,
        ``shape (..., n_chan, n_freq, n_frames)``

    Returns
    ----------
    Y: torch.Tensor, ``shape (..., n_src, n_freq, n_frames)``
        The separated signal in STFT-domain

    References
    ----------
    .. [10] C. Boeddeker et al.,
        "Convolutive Transfer Function Invariant SDR training criteria for Multi-Channel Reverberant Speech Separation",
        ICASSP, 2021.

    .. [11] Mehrez Souden, Jacob Benesty, and Sofiene Affes,
        "On optimal frequency-domain multichannel linear filtering for noise reduction",
        IEEE Trans. on audio, speech, and lang. process., 2009.

    """

    def __init__(
        self,
        mask_model: torch.nn.Module,
        ref_mic: Optional[int] = 0,
        eps: Optional[float] = 1e-5,
        mvdr_type: Optional[str] = "rtf",
        n_power_iter: Optional[int] = None,
    ):
        super().__init__(
            mask_model,
            ref_mic=ref_mic,
            eps=eps,
        )

        self.mvdr_type = mvdr_type
        self.n_power_iter = n_power_iter

    def forward(
        self,
        X: torch.Tensor,
        mask_model: Optional[torch.nn.Module] = None,
        ref_mic: Optional[int] = None,
        eps: Optional[float] = None,
        mvdr_type: Optional[str] = None,
        n_power_iter: Optional[int] = None,
    ):

        mask_model, ref_mic, eps, mvdr_type, n_power_iter = self._set_params(
            mask_model=mask_model,
            ref_mic=ref_mic,
            eps=eps,
            mvdr_type=mvdr_type,
            n_power_iter=n_power_iter,
        )

        # compute the masks (..., n_src, n_masks, n_freq, n_frames)
        masks = mask_model(X[..., ref_mic, :, :])

        n_masks = masks.shape[-3]
        masks = [masks[..., i, :, :] for i in range(n_masks)]

        if n_masks == 2:
            # compute the covariance matrices
            R_tgt, R_noise = [
                torch.einsum("...sfn,...cfn,...dfn->...sfcd", mask, X, X.conj())
                for mask in masks
            ]

            if mvdr_type == "rtf":
                # compute the relative transfer function
                rtf = compute_mvdr_rtf_eigh(
                    R_tgt,
                    R_noise,
                    ref_mic=ref_mic,
                    power_iterations=n_power_iter,
                )

                # compute the beamforming weights with rtf
                # shape (..., n_freq, n_chan)
                bf = compute_mvdr_bf(R_noise, rtf, eps=eps)

            elif mvdr_type == "scm":
                # compute the beamforming weights directly with spatial covariance matrices (scm)
                bf = compute_mvdr_bf2(R_tgt, R_noise, ref_mic=ref_mic, eps=eps)

        elif n_masks == 3:
            # if two noise masks are estimated as in [C. Boeddeker+, 2021]
            # compute the covariance matrices
            R_tgt, R_noise, R_noise_2 = [
                torch.einsum("...sfn,...cfn,...dfn->...sfcd", mask, X, X.conj())
                for mask in masks
            ]

            # compute the relative transfer function
            rtf = compute_mvdr_rtf_eigh(
                R_tgt,
                R_noise_2,
                ref_mic=ref_mic,
                power_iterations=n_power_iter,
            )

            # compute the beamforming weights
            # shape (..., n_freq, n_chan)
            bf = compute_mvdr_bf(R_noise, rtf, eps=eps)

        # compute output
        Y = torch.einsum("...cfn,...sfc->...sfn", X, bf.conj())

        return Y


class MWFBeamformer(BFBase):
    """
    Implementation of MWF beamformer described in [12]_.
    This class is basically assumes DNN-based beamforming.

    Parameters
    ----------
    mask_model: torch.nn.Module
        A function that is given one spectrogram and returns 2 masks
        of the same size as the input.
    ref_mic: int, optional
        Reference channel (default: ``0``)
    eps: float, optional
        A small constant to make divisions
        and the like numerically stable (default:``1e-5``).
    time_invariant: bool, optional
        If set to ``True``, this flag indicates that we want to use the
        time-invariant version of MWF.  If set to ``False``, the
        time-varying MWF is used instead (default: ``True``).

    Methods
    --------
    forward(X, mask_model = None, ref_mic = None, eps = None, time_invariant = None)

    Parameters
    ----------
    X: torch.Tensor
        The input mixture in STFT-domain,
        ``shape (..., n_chan, n_freq, n_frames)``

    Returns
    ----------
    Y: torch.Tensor, ``shape (..., n_src, n_freq, n_frames)``
        The separated signal in STFT-domain

    References
    ----------
    .. [12] Y. Masuyama et al.,
        "Consistency-aware multi-channel speech enhancement using deep neural networks",
        ICASSP, 2020.

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
            mask_model=mask_model,
            ref_mic=ref_mic,
            eps=eps,
            time_invariant=time_invariant,
        )

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
            Y = torch.einsum("...cfn,...sfc->...sfn", X, bf.conj())

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
            Y = torch.einsum("...cfn,...sfnc->...sfn", X, bf.conj())

        return Y


class GEVBeamformer(BFBase):

    """
    Implementation of GEV beamformer.
    This class is basically assumes DNN-based beamforming.

    Parameters
    ----------
    mask_model: torch.nn.Module
        A function that is given one spectrogram and returns 2 masks
        of the same size as the input.
    ref_mic: int, optional
        Reference channel (default: ``0``)
    eps: float, optional
        A small constant to make divisions
        and the like numerically stable (default:``1e-5``).

    Methods
    --------
    forward(X, mask_model = None, ref_mic = None, eps = None)

    Parameters
    ----------
    X: torch.Tensor
        The input mixture in STFT-domain,
        ``shape (..., n_chan, n_freq, n_frames)``

    Returns
    ----------
    Y: torch.Tensor, ``shape (..., n_src, n_freq, n_frames)``
        The separated signal in STFT-domain

    """

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
            mask_model=mask_model,
            ref_mic=ref_mic,
            eps=eps,
        )

        # Compute the mask
        masks = mask_model(X[..., ref_mic, :, :])

        covmat_masks = [masks[..., i, :, :] for i in range(2)]

        # compute the covariance matrices
        R_target, R_noise = [
            torch.einsum("...sfn,...cfn,...dfn->...sfcd", mask, X, X.conj())
            for mask in covmat_masks
        ]

        # Compute the beamformers
        gev_bf = compute_gev_bf(R_target, R_noise, ref_mic=ref_mic)

        # Separation
        # (batch, sources, freq, frames)
        Y = bmm(gev_bf, X.transpose(-3, -2)).transpose(-3, -2)

        return Y
