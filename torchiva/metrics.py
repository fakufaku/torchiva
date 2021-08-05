from typing import List, Optional, Tuple

import numpy as np
import torch as pt
from scipy.optimize import linear_sum_assignment


def _normalize(x: pt.Tensor, eps: Optional[float] = 1e-6, dim: Optional[int] = None):
    x = x / pt.clamp(x.norm(dim=dim, keepdim=True), min=eps)
    return x


def _db_clamp_eps(db_max: float):
    e = 10.0 ** (-db_max / 10.0)
    eps = e / (1.0 + e)
    return eps


def si_sdr(
    reference_signals: pt.Tensor,
    estimated_signals: pt.Tensor,
    db_clamp: Optional[float] = 60,
) -> Tuple[pt.Tensor, pt.Tensor]:
    """
    Compute the Scaled Invariant Signal-to-Distortion Ration (SI-SDR) and related
    measures according to [1]_.

    .. [1] J. Le Roux, S. Wisdom, H. Erdogan, J. R. Hershey, "SDR - half-baked or well
        done?", 2018, https://arxiv.org/abs/1811.02508

    Parameters
    ----------
    reference_signals: torch.Tensor (..., n_channels, n_samples)
        The reference clean signals
    estimated_signal: torch.Tensor (..., n_channels, n_samples)
        The signals to evaluate
    db_clamp: float
        Restrict the output to be between -db_clamp dB and db_clamp dB

    Returns
    -------
    SI-SDR: torch.Tensor (..., n_channels)
        Signal-to-Distortion Ratio
    p_opts: torch.Tensor (..., n_channels)
        The optimal ordering of the channels
    """

    # normalize
    reference_signals = _normalize(reference_signals, dim=-1)
    estimated_signals = _normalize(estimated_signals, dim=-1)

    # compute the squared coherence
    coh = pt.einsum("...ct,...dt->...cd", reference_signals, estimated_signals)
    coh = coh ** 2

    # clamp within desired decibel range
    eps = _db_clamp_eps(db_clamp)
    coh = pt.clamp(coh, min=eps, max=1 - eps)
    ratio = (1 - coh) / coh

    # apply the SDR mapping
    si_sdr = -10.0 * pt.log10(ratio)

    # solve the permutation
    si_sdr, _, __, p_opts = _solve_permutation(si_sdr, si_sdr, si_sdr)

    return si_sdr, p_opts


def si_bss_eval(
    reference_signals: pt.Tensor,
    estimated_signals: pt.Tensor,
    scaling: Optional[bool] = True,
) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
    """
    Compute the Scaled Invariant Signal-to-Distortion Ration (SI-SDR) and related
    measures according to [1]_.

    .. [1] J. Le Roux, S. Wisdom, H. Erdogan, J. R. Hershey, "SDR - half-baked or well
        done?", 2018, https://arxiv.org/abs/1811.02508

    Parameters
    ----------
    reference_signals: torch.Tensor (..., n_channels, n_samples)
        The reference clean signals
    estimated_signal: torch.Tensor (..., n_channels, n_samples)
        The signals to evaluate
    scaling: bool
        Flag that indicates whether we want to use the scale invariant (True)
        or scale dependent (False) method

    Returns
    -------
    SDR: torch.Tensor (..., n_channels)
        Signal-to-Distortion Ratio
    SIR: torch.Tensor (..., n_channels)
        Signal-to-Interference Ratio
    SAR: torch.Tensor (..., n_channels)
        Signal-to-Artefact Ratio
    p_opts: torch.Tensor (..., n_channels)
        The optimal ordering of the channels
    """

    # invert the last two dimensions
    # -> shape == (..., n_samples, n_channels)
    reference_signals = reference_signals.transpose(-2, -1)
    estimated_signals = estimated_signals.transpose(-2, -1)

    # now compute the shapes
    shape_batch = estimated_signals.shape[:-2]
    n_samples_est, n_chan_est = estimated_signals.shape[-2:]
    n_samples_ref, n_chan_ref = reference_signals.shape[-2:]

    # Equalize the length of the signals
    m = min(n_samples_est, n_samples_ref)
    reference_signals = reference_signals[..., :m, :]
    estimated_signals = estimated_signals[..., :m, :]

    # pre-compute the covariance matrix of the reference signals
    # shape == (..., n_chan_ref, n_chan_ref)
    Rss = pt.matmul(reference_signals.transpose(-2, -1), reference_signals)

    # dirty hack: sometimes there is a zero reference signal in the dataset, which makes
    # the function hard fail when trying to invert Rss
    # As a countermeasure, we add a small quantity in the diagonal if the diagonal
    # component is zero
    for i in range(n_chan_ref):
        Rss[..., i, i].clamp_(min=1e-5)

    output_shape = shape_batch + (n_chan_ref, n_chan_est)
    SDR = pt.zeros(output_shape).type_as(reference_signals)
    SIR = pt.zeros(output_shape).type_as(reference_signals)
    SAR = pt.zeros(output_shape).type_as(reference_signals)

    for r in range(n_chan_ref):
        for e in range(n_chan_est):
            SDR[..., r, e], SIR[..., r, e], SAR[..., r, e] = _compute_measures(
                estimated_signals[..., e], reference_signals, Rss, r, scaling=scaling
            )

    return _solve_permutation(SDR, SIR, SAR)


def _compute_measures(
    estimated_signal: pt.Tensor,
    reference_signals: pt.Tensor,
    Rss: pt.Tensor,
    j: int,
    scaling: Optional[bool] = True,
) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor]:
    """
    Compute the Scale Invariant SDR and other metrics

    The original implementation was provided by Johnathan Le Roux
    [here](https://github.com/sigsep/bsseval/issues/3)
    It was rewritten in pytorch by Robin Scheibler

    Parameters
    ----------
    estimated_signal: torch.Tensor (..., n_samples)
        The signals to evaluate
    reference_signals: torch.Tensor (..., n_samples, n_channels)
        The reference clean signals
    Rss: ndarray (..., n_channels, n_channels)
        The covariance matrix of the reference signals
    j: int
        The index of the source to evaluate
    scaling: bool
        Flag that indicates whether we want to use the scale invariant (True)
        or scale dependent (False) method
    """
    this_s = reference_signals[..., :, j, None]
    estimated_signal = estimated_signal[..., None]

    if scaling:
        # get the scaling factor for clean sources
        a = pt.sum(this_s * estimated_signal, dim=-2) / Rss[..., j, j, None]
        a = a[..., None]
    else:
        a = 1

    e_true = a * this_s
    e_res = estimated_signal - e_true

    Sss = (e_true.square()).sum(dim=(-2, -1))
    Snn = (e_res.square()).sum(dim=(-2, -1))

    SDR = 10 * pt.log10((Sss / Snn.clamp(min=1e-5)).clamp(min=1e-5))

    # Get the SIR
    Rsr = pt.matmul(reference_signals.transpose(-2, -1), e_res)
    b = pt.linalg.solve(Rss, Rsr)  # caution: order of lhs and rhs is reverse in torch

    e_interf = pt.matmul(reference_signals, b)
    e_artif = e_res - e_interf

    SIR = 10 * pt.log10(Sss / (1e-5 + (e_interf ** 2).sum(dim=(-2, -1))))
    SAR = 10 * pt.log10(Sss / (1e-5 + (e_artif ** 2).sum(dim=(-2, -1))))

    return SDR, SIR, SAR


def _solve_permutation(
    SDR: pt.Tensor, SIR: pt.Tensor, SAR: pt.Tensor
) -> Tuple[pt.Tensor, pt.Tensor, pt.Tensor, pt.Tensor]:
    """
    Solve the permutation in numpy for now
    """

    b_shape = SIR.shape[:-2]
    n_chan_ref, n_chan_est = SIR.shape[-2:]
    n_chan_out = min(n_chan_ref, n_chan_est)

    if n_chan_ref > n_chan_est:
        SDR = SDR.transpose(-2, -1)
        SIR = SIR.transpose(-2, -1)
        SAR = SAR.transpose(-2, -1)

    SIR_npy = SIR.cpu().detach().numpy()

    SDR_out = SDR.new_zeros(b_shape + (n_chan_out,))
    SIR_out = SIR.new_zeros(b_shape + (n_chan_out,))
    SAR_out = SAR.new_zeros(b_shape + (n_chan_out,))

    p_opts = np.zeros(b_shape + (n_chan_out,), dtype=np.int64)
    for m in np.ndindex(b_shape):
        dum, p_opt = _linear_sum_assignment_with_inf(-SIR_npy[m])
        SDR_out[m] = SDR[m + (dum, p_opt)]
        SIR_out[m] = SIR[m + (dum, p_opt)]
        SAR_out[m] = SAR[m + (dum, p_opt)]
        p_opts[m] = p_opt

    p_opts = pt.from_numpy(p_opts).to(SDR_out.device)
    return SDR_out, SIR_out, SAR_out, p_opts


def _linear_sum_assignment_with_inf(
    cost_matrix: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Solves the permutation problem efficiently via the linear sum
    assignment problem.
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.linear_sum_assignment.html
    This implementation was proposed by @louisabraham in
    https://github.com/scipy/scipy/issues/6900
    to handle infinite entries in the cost matrix.
    """
    cost_matrix = np.asarray(cost_matrix)
    min_inf = np.isneginf(cost_matrix).any()
    max_inf = np.isposinf(cost_matrix).any()
    if min_inf and max_inf:
        raise ValueError("matrix contains both inf and -inf")

    if min_inf or max_inf:
        cost_matrix = cost_matrix.copy()
        values = cost_matrix[~np.isinf(cost_matrix)]
        m = values.min()
        M = values.max()
        n = min(cost_matrix.shape)
        # strictly positive constant even when added
        # to elements of the cost matrix
        positive = n * (M - m + np.abs(M) + np.abs(m) + 1)
        if max_inf:
            place_holder = (M + (n - 1) * (M - m)) + positive
        if min_inf:
            place_holder = (m + (n - 1) * (m - M)) - positive

        cost_matrix[np.isinf(cost_matrix)] = place_holder

    return linear_sum_assignment(cost_matrix)
