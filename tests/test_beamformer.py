import torchiva
import fast_bss_eval
import torch
import torchaudio
import json
import pytest
from pathlib import Path

from torchiva.linalg import eigh, solve_loaded, bmm, hermite, multiply

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

ref_mic=0

with open(Path("wsj1_6ch") / "dev93" / "mixinfo_noise.json") as f:
    mixinfo = json.load(f)

info = mixinfo["00001"]

mix, fs = torchaudio.load(Path("wsj1_6ch") / (Path("").joinpath(*Path(info['wav_mixed_noise_reverb']).parts[-4:])))

ref1, fs = torchaudio.load(Path("wsj1_6ch") / (Path("").joinpath(*Path(info['wav_dpath_image_reverberant'][0]).parts[-4:])))
ref2, fs = torchaudio.load(Path("wsj1_6ch") / (Path("").joinpath(*Path(info['wav_dpath_image_reverberant'][1]).parts[-4:])))
ref = torch.stack((ref1[ref_mic], ref2[ref_mic]),dim=0)


def compute_covariance_matrix(X):
    R = torch.einsum('...mft, ...nft->...fmnt', X, X.conj()).mean(dim=-1)
    return R

def mse(ref, y):
    return torch.mean(abs(ref-y)**2)

@pytest.mark.parametrize(
    "n_fft",
    [
        (4096),
    ],
)
def test_beamformers(n_fft):

    global mix, ref, ref1, ref2, ref_mic

    stft = torchiva.STFT(
        n_fft=n_fft,
    )

    X = stft(mix)
    REF = stft(torch.stack((ref1,ref2),dim=-3))

    # (n_src, n_freq, n_chan, n_chan)
    R_tgt = compute_covariance_matrix(REF)
    R_noise = compute_covariance_matrix(X-REF)

    # -------------------------
    # MWF part
    mwf = torchiva.compute_mwf_bf(R_tgt, R_noise, ref_mic=ref_mic)
    Y = torch.einsum("...cfn,...sfc->...sfn", X, mwf.conj())
    y = stft.inv(Y)

    m = min(ref.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])
    print(f"\nMWF     SDR", sdr)


    # -------------------------
    # MVDR part 
    rtf = torchiva.compute_mvdr_rtf_eigh(R_tgt, R_noise, ref_mic=ref_mic, power_iterations=15)
    mvdr = torchiva.compute_mvdr_bf(R_noise, rtf)
    Y = torch.einsum("...cfn,...sfc->...sfn", X, mvdr.conj())
    y = stft.inv(Y)
    
    m = min(ref.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])
    print(f"MVDRpi  SDR", sdr)


    rtf = torchiva.compute_mvdr_rtf_eigh(R_tgt, R_noise, ref_mic=ref_mic, power_iterations=None)
    mvdr = torchiva.compute_mvdr_bf(R_noise, rtf)
    Y = torch.einsum("...cfn,...sfc->...sfn", X, mvdr.conj())
    y = stft.inv(Y)
    
    m = min(ref.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])
    print(f"MVDRgev SDR", sdr)


    # -------------------------
    # MVDR2 part
    mvdr2 = torchiva.compute_mvdr_bf2(R_tgt, R_noise)
    Y = torch.einsum("...cfn,...sfc->...sfn", X, mvdr2.conj())
    y = stft.inv(Y)
    
    m = min(ref.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])
    print(f"MVDRscm SDR", sdr)


    # -------------------------
    # GEV part
    gev = torchiva.compute_gev_bf(R_tgt, R_noise, ref_mic=ref_mic)
    Y = bmm(gev, X.transpose(-3, -2)).transpose(-3, -2)
    y = stft.inv(Y)

    m = min(ref.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])
    print(f"GEV     SDR", sdr)