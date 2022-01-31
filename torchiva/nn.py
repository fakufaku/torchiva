import enum
from typing import List, Optional, Dict, Union

import torch as pt
from torch import nn

from .auxiva_iss import auxiva_iss
from .base import SourceModelBase
from .five import five
from .linalg import bmm, eigh, hermite, multiply
from .metrics import si_bss_eval
from .models import LaplaceModel
from .overiva import overiva
from .scaling import minimum_distortion, minimum_distortion_l2_phase, projection_back
from .mvdr import compute_mvdr_rtf_eigh, compute_mvdr_bf
from .mwf import compute_mwf_bf
from .stft import STFT
from .utils import module_from_config


class SepAlgo(enum.Enum):
    AUXIVA_ISS = "auxiva-iss"
    AUXIVA_IP2 = "auxiva-ip2"
    OVERIVA_IP = "overiva-ip"
    FIVE = "five"


class Separator(nn.Module):
    def __init__(
        self,
        n_fft: int,
        n_src: Optional[int] = None,
        algo: Optional[SepAlgo] = SepAlgo.AUXIVA_ISS,
        hop_length: Optional[int] = None,
        window: Optional[int] = None,
        source_model: Optional[SourceModelBase] = None,
        n_iter: Optional[int] = 10,
        ref_mic: Optional[int] = 0,
        mdp_p: Optional[float] = None,
        mdp_q: Optional[float] = None,
        proj_back: Optional[bool] = False,
        mdp_phase: Optional[bool] = False,
        mdp_model: Optional[nn.Module] = None,
        postfilter_model: Optional[nn.Module] = None,
        algo_kwargs: Optional[Dict] = None,
        checkpoints: Optional[List[int]] = None,
    ):

        super().__init__()

        if source_model is None:
            self.source_model = LaplaceModel()
        else:
            if isinstance(source_model, dict):
                self.source_model = module_from_config(**source_model)
            else:
                self.source_model = source_model

        self.n_src = n_src
        self.algo = algo
        if algo_kwargs is None:
            self.algo_kwargs = {}
        else:
            self.algo_kwargs = algo_kwargs

        # other attributes
        self.ref_mic = ref_mic
        self.n_iter = n_iter
        self.mdp_p = mdp_p
        self.mdp_q = mdp_q
        self.proj_back = proj_back
        self.mdp_phase = mdp_phase

        if isinstance(mdp_model, dict):
            self.mdp_model = module_from_config(**mdp_model)
        else:
            self.mdp_model = mdp_model

        # we can also optionnaly use a post-filter masking model
        if isinstance(postfilter_model, dict):
            self.postfilter_model = module_from_config(**postfilter_model)
        else:
            self.postfilter_model = postfilter_model

        # the stft engine
        self.stft = STFT(n_fft, hop_length=hop_length, window=window)

        # in some cases, we want to instrument
        self.checkpoints = checkpoints
        self.saved_checkpoints = None

    @property
    def algo(self):
        return self._algo.value

    @algo.setter
    def algo(self, val):
        self._algo = SepAlgo(val)

        if val == SepAlgo.FIVE and self.n_src is not None and self.n_src > 1:
            import warnings

            warnings.warn(
                "Algorithm FIVE can only extract one source. Parameter n_src ignored."
            )

    def postfilter(self, Y):
        if self.postfilter_model is None:
            return Y
        else:
            mask = self.postfilter_model(Y[..., :, :])
            mask = pt.mean(mask[..., 0, 0, :, :], dim=-1, keepdim=True)
            return mask * Y

    def scale(self, Y, X):

        if self.proj_back:
            Y = projection_back(Y, X[..., self.ref_mic, :, :])
        elif self.mdp_phase:
            Y = minimum_distortion_l2_phase(Y, X[..., self.ref_mic, :, :],)
        elif self.mdp_model is not None or self.mdp_p is not None:
            Y = minimum_distortion(
                Y,
                X[..., self.ref_mic, :, :],
                p=self.mdp_p,
                q=self.mdp_q,
                model=self.mdp_model,
                max_iter=1,
            )
        else:
            pass

        return Y

    def forward(self, x, n_iter=None, reset=True):
        if n_iter is None:
            n_iter = self.n_iter

        if hasattr(self.source_model, "reset") and reset:
            # this is required for models with internal state, such a ILRMA
            self.source_model.reset()

        n_chan, n_samples = x.shape[-2:]

        if self.n_src is None:
            n_src = n_chan
        else:
            n_src = self.n_src

        assert n_chan >= n_src, (
            "The number of channel should be larger or equal to "
            "the number of sources to separate."
        )

        # prepare a list for checkpoints just in case
        saved_checkpoints = []

        # STFT
        X = self.stft(x)  # copy for back projection (numpy/torch compatible)

        # Separation
        if self._algo == SepAlgo.AUXIVA_ISS:
            Y = auxiva_iss(
                X,
                n_iter=n_iter,
                model=self.source_model,
                two_chan_ip2=False,
                checkpoints_iter=self.checkpoints,
                checkpoints_list=saved_checkpoints,
                **self.algo_kwargs,
            )

        elif self._algo == SepAlgo.AUXIVA_IP2:
            Y = auxiva_iss(
                X,
                n_iter=n_iter,
                model=self.source_model,
                two_chan_ip2=True,
                checkpoints_iter=self.checkpoints,
                checkpoints_list=saved_checkpoints,
                **self.algo_kwargs,
            )

        elif self._algo == SepAlgo.OVERIVA_IP:
            Y = overiva(
                X,
                n_src=self.n_src,
                n_iter=n_iter,
                model=self.source_model,
                checkpoints_iter=self.checkpoints,
                checkpoints_list=saved_checkpoints,
                **self.algo_kwargs,
            )

        elif self._algo == SepAlgo.FIVE:
            Y = five(
                X,
                n_iter=n_iter,
                model=self.source_model,
                ref_mic=self.ref_mic,
                checkpoints_iter=self.checkpoints,
                checkpoints_list=saved_checkpoints,
                **self.algo_kwargs,
            )
        else:
            raise NotImplementedError

        # Solve scale ambiguity
        Y = self.scale(Y, X)

        # Post-filtering
        Y = self.postfilter(Y)

        # iSTFT
        y = self.stft.inv(Y)  # (n_samples, n_channels)

        # in case we separated too many sources, select those that have most energy
        if self.n_src is not None and y.shape[-2] > self.n_src:
            y = most_energetic(y, num=self.n_src, dim=-2, dim_reduc=-1)

        # post-process the checkpoints if required
        if self.checkpoints is not None:
            self.saved_checkpoints = {
                "iter": self.checkpoints[: len(saved_checkpoints)],
                "checkpoints": saved_checkpoints,
                "ref": X,
            }

        if y.shape[-1] < x.shape[-1]:
            y = pt.cat(
                (y, y.new_zeros(y.shape[:-1] + (x.shape[-1] - y.shape[-1],))), dim=-1
            )
        elif y.shape[-1] > x.shape[-1]:
            y = y[..., : x.shape[-1]]

        return y

    def process_checkpoints(self):
        """
        We adjust scale and inv-STFT all the saved spectrograms
        """
        if self.saved_checkpoints is None:
            return

        # concatenate and broad cast to the same shape
        X = self.saved_checkpoints["ref"]
        Ys = pt.cat(
            [Y[None, ...] for Y in self.saved_checkpoints["checkpoints"]], dim=0
        )
        X, Ys = pt.broadcast_tensors(X[None, ...], Ys)

        # adjust scale
        Ys = self.scale(Ys, X)

        # Post-filtering
        Ys = self.postfilter(Ys)

        # transform to time-domain
        ys = self.stft.inv(Ys)

        ret = {"iter": self.saved_checkpoints["iter"], "signals": ys}
        self.saved_checkpoints = None

        return ret


class MaskGEVBeamformer(nn.Module):
    def __init__(
        self,
        n_fft: int,
        hop_length: Optional[int] = None,
        window: Optional[int] = None,
        mask_model: Optional[pt.nn.Module] = None,
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

    def _compute_gev_bf(self, X, mask):
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
        rhs = pt.eye(eigvec.shape[-1], dtype=eigvec.dtype, device=eigvec.device)
        rhs = rhs[:, -1, None]
        steering_vector = pt.linalg.solve(hermite(eigvec), rhs)
        scale = steering_vector[..., self.ref_mic, None, :]

        # (batch, freq, sources, channels)
        gev_bf = pt.conj(eigvec[..., -1].transpose(-3, -2)) * scale

        return gev_bf

    def forward(self, x):

        # STFT (batch, channels, freq, frames)
        X = self.stft(x)

        # Compute the mask
        mask = self.mask_model(X[..., self.ref_mic, :, :])

        # Compute the beamformers
        gev_bf = self._compute_gev_bf(X, mask)

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


class MaskSeparator(nn.Module):
    """
    Although this one takes multiple channels as inputs, it only
    uses one for separation
    """

    def __init__(
        self,
        n_fft: int,
        hop_length: Optional[int] = None,
        window: Optional[int] = None,
        mask_model: Optional[pt.nn.Module] = None,
        ref_mic: Optional[int] = 0,
    ):

        super().__init__()

        self.mask_model = mask_model
        if isinstance(mask_model, dict):
            self.mask_model = module_from_config(**mask_model)
        else:
            self.mask_model = mask_model

        # other attributes
        self.ref_mic = ref_mic

        # the stft engine
        self.stft = STFT(n_fft, hop_length=hop_length, window=window)

    def forward(self, x):

        # STFT (batch, channels, freq, frames)
        X = self.stft(x)

        # Compute the mask
        mask = self.mask_model(X[..., self.ref_mic, :, :])

        # Separation
        # (batch, sources, freq, frames)
        Y = multiply(X[..., self.ref_mic : self.ref_mic + 1, :, :], mask[..., :, :, :])

        # iSTFT
        y = self.stft.inv(Y)  # (n_samples, n_channels)

        return y


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
        mask_model: Optional[Union[pt.nn.Module, Dict]] = None,
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
            pt.einsum("...sfn,...cfn,...dfn->...sfcd", mask, X, X.conj())
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
        X = pt.einsum("...cfn,...sfc->...sfn", X, bf.conj())

        # add back DC offset
        pad_shape = X.shape[:-2] + (1,) + X.shape[-1:]
        X = pt.cat((X.new_zeros(pad_shape), X), dim=-2)

        y = self.stft.inv(X)

        if y.shape[-1] < x.shape[-1]:
            y = pt.cat(
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
        mask_model: Optional[Union[pt.nn.Module, Dict]] = None,
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
            pt.einsum("...sfn,...cfn,...dfn->...sfcd", mask, X, X.conj())
            for mask in covmat_masks
        ]

        if self.time_invariant:

            # compute the beamforming weights
            # shape (..., n_freq, n_chan)
            bf = compute_mwf_bf(R_tgt, R_noise, ref_mic=self.ref_mic)

            # compute output
            X = pt.einsum("...cfn,...sfc->...sfn", X, bf.conj())

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
            X = pt.einsum("...cfn,...sfnc->...sfn", X, bf.conj())

        # add back DC offset
        pad_shape = X.shape[:-2] + (1,) + X.shape[-1:]
        X = pt.cat((X.new_zeros(pad_shape), X), dim=-2)

        y = self.stft.inv(X)

        if y.shape[-1] < x.shape[-1]:
            y = pt.cat(
                (y, y.new_zeros(y.shape[:-1] + (x.shape[-1] - y.shape[-1],))), dim=-1
            )
        elif y.shape[-1] > x.shape[-1]:
            y = y[..., : x.shape[-1]]

        return y


def sdr_loss(
    estimated: pt.Tensor, reference: pt.Tensor, reduce: Optional[bool] = True,
):
    """
    Negative signal-to-distortion ratio loss. This loss is permuatation invariant.

    Parameters
    ----------
    estimated: torch.Tensor, shape (n_batch, n_channels, n_samples)
        The estimated signals
    reference: torch.Tensor, shape (n_batch, n_channels, n_samples_ref)
        The reference signals, when the number of samples of estimated and reference
        differ, they are truncated to ``min(n_samples, n_samples_ref)``
    reduce: bool, optional
        By default, the loss is averaged over all signals in the batch, but when
        set to ``False``, the individual values are returned

    Returns
    -------
    neg_sdr: torch.Tensor, shape (1) or (n_batch, n_channels)
        The negative SDR of the estimated signals
    """

    # scale invaliant metric
    sdr, sir, sar, perm = si_bss_eval(reference, estimated)

    # we only care about the SDR
    if reduce:
        return -sdr.mean()
    else:
        return -sdr


def most_energetic(
    x: pt.Tensor, num: int, dim: Optional[int] = -2, dim_reduc: Optional[int] = -1
):
    """
    Selects the `num` indices with most power

    Parameters
    ----------
    x: torch.Tensor
        The input tensor
    num: int
    dim:
        The axis where the selection should occur
    dim_reduc:
        The axis where to perform the reduction
    """

    power = x.abs().square().mean(axis=dim_reduc, keepdim=True)

    index = pt.argsort(power.transpose(dim, -1), axis=-1, descending=True)
    index = index[..., :num]

    # need to broadcast to target size
    x_tgt = x.transpose(dim, -1)[..., :num]
    _, index = pt.broadcast_tensors(x_tgt, index)

    # reorder index axis
    index = index.transpose(dim, -1)

    return pt.gather(x, dim=dim, index=index)
