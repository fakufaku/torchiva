import torchiva
import fast_bss_eval
import torch
import torchaudio
import json
import pytest
from pathlib import Path

from torchiva.linalg import eigh, solve_loaded, bmm, hermite, multiply
import torchiva.beamformer

from examples.samples.read_samples import read_samples

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

ref_mic = 0

mix, ref_wet, ref_dry = read_samples()

snr = 40
mix_std = torch.std(mix)
noise_std = mix_std * 10 ** (-snr / 20)  # SNR = 20

mix = mix[0] + mix.new_zeros(mix.shape).normal_() * noise_std
ref = ref_wet[0, :, ref_mic, :]
ref1 = ref_wet[0, 0, :, :]
ref2 = ref_wet[0, 1, :, :]


def compute_covariance_matrix(X):
    R = torch.einsum("...mft, ...nft->...fmnt", X, X.conj()).mean(dim=-1)
    return R


def mse(ref, y):
    return torch.mean(abs(ref - y) ** 2)


def compute_stft_and_covmats(mix, ref1, ref2, n_fft):
    stft = torchiva.STFT(n_fft=n_fft)

    X = stft(mix)
    REF = stft(torch.stack((ref1, ref2), dim=-3))

    # (n_src, n_freq, n_chan, n_chan)
    R_tgt = compute_covariance_matrix(REF)
    R_noise = compute_covariance_matrix(X - REF)

    return X, R_tgt, R_noise, stft


@pytest.mark.parametrize(
    "n_fft, tol_db",
    [
        (4096, 18),
    ],
)
def test_mwf_beamformer(n_fft, tol_db):

    global mix, ref1, ref2, ref
    X, R_tgt, R_noise, stft = compute_stft_and_covmats(mix, ref1, ref2, n_fft)

    # -------------------------
    # MWF part
    mwf = torchiva.beamformer.compute_mwf_bf(R_tgt, R_noise, ref_mic=ref_mic)
    Y = torch.einsum("...cfn,...sfc->...sfn", X, mwf.conj())
    y = stft.inv(Y)

    m = min(ref.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])
    print(f"\nMWF     SDR", sdr)

    if tol_db is not None:
        assert sdr.mean() > tol_db


@pytest.mark.parametrize(
    "n_fft, tol_db",
    [
        (4096, 10),
    ],
)
def test_mvdr_pwr_it_beamformer(n_fft, tol_db):

    global mix, ref1, ref2, ref
    X, R_tgt, R_noise, stft = compute_stft_and_covmats(mix, ref1, ref2, n_fft)

    # -------------------------
    # MVDR part
    rtf = torchiva.beamformer.compute_mvdr_rtf_eigh(
        R_tgt, R_noise, ref_mic=ref_mic, power_iterations=15
    )
    mvdr = torchiva.beamformer.compute_mvdr_bf(R_noise, rtf)
    Y = torch.einsum("...cfn,...sfc->...sfn", X, mvdr.conj())
    y = stft.inv(Y)

    m = min(ref.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])
    print(f"MVDRpi  SDR", sdr)

    if tol_db is not None:
        assert sdr.mean() > tol_db


@pytest.mark.parametrize(
    "n_fft, tol_db",
    [
        (4096, 15),
    ],
)
def test_mvdr_gev_beamformer(n_fft, tol_db):

    global mix, ref1, ref2, ref
    X, R_tgt, R_noise, stft = compute_stft_and_covmats(mix, ref1, ref2, n_fft)

    rtf = torchiva.beamformer.compute_mvdr_rtf_eigh(
        R_tgt, R_noise, ref_mic=ref_mic, power_iterations=None
    )
    mvdr = torchiva.beamformer.compute_mvdr_bf(R_noise, rtf)
    Y = torch.einsum("...cfn,...sfc->...sfn", X, mvdr.conj())
    y = stft.inv(Y)

    m = min(ref.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])
    print(f"MVDRgev SDR", sdr)

    if tol_db is not None:
        assert sdr.mean() > tol_db


@pytest.mark.parametrize(
    "n_fft, tol_db",
    [
        (4096, 18),
    ],
)
def test_mvdr2_beamformer(n_fft, tol_db):

    global mix, ref1, ref2, ref
    X, R_tgt, R_noise, stft = compute_stft_and_covmats(mix, ref1, ref2, n_fft)

    # -------------------------
    # MVDR2 part
    mvdr2 = torchiva.beamformer.compute_mvdr_bf2(R_tgt, R_noise)
    Y = torch.einsum("...cfn,...sfc->...sfn", X, mvdr2.conj())
    y = stft.inv(Y)

    m = min(ref.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])
    print(f"MVDRscm SDR", sdr)

    if tol_db is not None:
        assert sdr.mean() > tol_db


@pytest.mark.parametrize(
    "n_fft, tol_db",
    [
        (4096, 18),
    ],
)
def test_gev_beamformer(n_fft, tol_db):

    global mix, ref1, ref2, ref
    X, R_tgt, R_noise, stft = compute_stft_and_covmats(mix, ref1, ref2, n_fft)

    # -------------------------
    # GEV part
    gev = torchiva.beamformer.compute_gev_bf(R_tgt, R_noise, ref_mic=ref_mic)
    Y = bmm(gev, X.transpose(-3, -2)).transpose(-3, -2)
    y = stft.inv(Y)

    m = min(ref.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])
    print(f"GEV     SDR", sdr)

    if tol_db is not None:
        assert sdr.mean() > tol_db


if __name__ == "__main__":
    test_mwf_beamformer(4096, None)
    test_mvdr_pwr_it_beamformer(4096, None)
    test_mvdr_gev_beamformer(4096, None)
    test_mvdr2_beamformer(4096, None)
    test_gev_beamformer(4096, None)
