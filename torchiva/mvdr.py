import enum
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .linalg import eigh


def compute_covariance_matrices(X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Parameters
    ----------
    X: torch.Tensor, (..., n_chan, n_freq, n_frames)
        The input STFT signals
    mask: torch.Tensor, (..., n_src, n_freq, n_frames)
        The masks

    Returns
    -------
    R: torch.Tensor, (..., n_src, n_freq, n_chan, n_chan)
        The weighted covariance matrices
    """
    R = torch.einsum("...sfn,...cfn,...dfn->...sfcd", mask, X, torch.conj(X))
    return R


def compute_relative_transfer_function(
    R_target: torch.Tensor,
    R_dist: torch.Tensor,
    ref_mic: Optional[int] = 0,
    n_power_iter: Optional[int] = None,
):
    """
    Parameters
    ----------
    X: torch.Tensor, (..., n_chan, n_freq, n_frames)
        The input STFT signals
    mask: torch.Tensor, (..., n_src, n_freq, n_frames)
        The masks

    Returns
    -------
    R: torch.Tensor, (..., n_src, n_freq, n_chan, n_chan)
        The weighted covariance matrices
    """

    V = torch.linalg.solve(R_dist, R_target)

    if n_power_iter is None:
        # use the regular thing
        eigenvalues, eigenvectors = eigh(A, B)
        v = eigenvectors[..., :, -1]

    else:

        U = torch.linalg.solve(B, A)
        # one-hot vector
        v = U.new_zeros(V.shape[:-2] + V.shape[-1:])
        v[..., ref_mic] = 1.0
        for epoch in range(n_power_iter):
            v = U @ v

    v = torch.ensum("...cd,...d->...c", R_dist, v)  # matrix-vector mult.
    v = v / v[..., ref_mic]  # make the TF relative

    return v


class MVDRBeamformer(nn.Module):
    """
    This is the neural network supported MVDR beamformer described in the paper

    C. Boeddeker et al., "CONVOLUTIVE TRANSFER FUNCTION INVARIANT SDR TRAINING
    CRITERIA FORMULTI-CHANNEL REVERBERANT SPEECH SEPARATION", Proc. ICASSP
    2021.
    """

    def __init__(
        self, ref_mic: Optional[int] = 0, use_n_power_iter: Optional[int] = None
    ):
        super().__init__()

        self.use_n_power_iter = use_n_power_iter

    def forward(self, X: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        batch_size = X.shape[:-3]
        n_chan, n_freq, n_frames = X.shape[-3:]
        n_src = mask.shape[-3] // 3

        # shape: (..., 3 * n_src, n_freq, n_chan, n_chan)
        R = compute_covariance_matrices(X, mask)

        # re-organize
        R = R.reshape(batch_size + (3, n_src, n_freq, n_chan, n_chan))
        R_target = R[..., 0, :, :, :, :]  # covariance of target
        R_dist_1 = R[..., 1, :, :, :, :]  # covariance of distortion
        R_dist_2 = R[..., 2, :, :, :, :]  # for dominant eigenvector estimation

        # transfer function
        v = compute_relative_transfer_function(
            R_target, R_dist_2, n_power_iter=self.use_n_power_iter, ref_mic=self.ref_mic
        )

        # beamforming weights
        Rinv_v = torch.linalg.solve(R_dist_1, v)
        denom = torch.einsum("...c,...c->...", torch.conj(v), Rinv_v)
        w = Rinv_v / denom[..., None]

        # now do the beamforming
        X = torch.einsum("...sfc,...cfn->...sfn", torch.conj(w), X)

        return X
