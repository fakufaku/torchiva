import torchiva
import fast_bss_eval
import torch
import torchaudio
import json
import pytest
from pathlib import Path

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

ref_mic=1

with open(Path("wsj1_6ch") / "dev93" / "mixinfo_noise.json") as f:
    mixinfo = json.load(f)

info = mixinfo["00224"]

mix, fs = torchaudio.load(Path("wsj1_6ch") / (Path("").joinpath(*Path(info['wav_mixed_noise_reverb']).parts[-4:])))

#if delay==0 and tap==0:
#    ref1, fs = torchaudio.load(Path("wsj1_6ch") / (Path("").joinpath(*Path(info['wav_dpath_image_reverberant'][0]).parts[-4:])))
#    ref2, fs = torchaudio.load(Path("wsj1_6ch") / (Path("").joinpath(*Path(info['wav_dpath_image_reverberant'][1]).parts[-4:])))
#else:
#    ref1, fs = torchaudio.load(Path("wsj1_6ch") / (Path("").joinpath(*Path(info['wav_dpath_image_anechoic'][0]).parts[-4:])))
#    ref2, fs = torchaudio.load(Path("wsj1_6ch") / (Path("").joinpath(*Path(info['wav_dpath_image_anechoic'][1]).parts[-4:])))

ref1, fs = torchaudio.load(Path("wsj1_6ch") / (Path("").joinpath(*Path(info['wav_dpath_image_anechoic'][0]).parts[-4:])))
ref2, fs = torchaudio.load(Path("wsj1_6ch") / (Path("").joinpath(*Path(info['wav_dpath_image_anechoic'][1]).parts[-4:])))
ref = torch.stack((ref1[ref_mic], ref2[ref_mic]),dim=0)


@pytest.mark.parametrize(
    "n_iter, delay, tap, n_chan, n_src, n_fft",
    [
        (50, 0, 0, 2, 2, 4096),
        (20, 0, 0, 6, 2, 4096),
        #(50, 1, 5, 2, 2, 1024),
        #(20, 1, 5, 6, 2, 1024),
    ],
)
def test_overtiss(n_iter, delay, tap, n_chan, n_src, n_fft):

    global mix, ref

    x = mix.clone()
    x = x[:n_chan]

    stft = torchiva.STFT(
        n_fft=n_fft,
    )

    overtiss = torchiva.OverISS_T(
        n_iter,
        n_taps=tap,
        n_delay=delay,
        n_src=2,
        model = torchiva.models.GaussModel(),
        proj_back_mic=ref_mic,
        use_dmc=False,
        eps=None,
    )

    X = stft(x)

    Y = overtiss(X)
    y = stft.inv(Y)

    m = min(ref.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])

    print(f"\nOverTISS  iter:{n_iter:.0f} delay:{delay:.0f} tap:{tap:.0f} n_chan:{n_chan:.0f} n_src:{n_src:.0f} SDR", sdr)


    # change source model in forward call
    #Y = overtiss(X, model=torchiva.models.NMFModel())
    #y = stft.inv(Y)

    #m = min(ref.shape[-1], y.shape[-1])
    #sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])

    #print(f"\nOverTISS  iter:{n_iter:.0f} delay:{delay:.0f} tap:{tap:.0f} n_chan:{n_chan:.0f} n_src:{n_src:.0f} SDR", sdr)


@pytest.mark.parametrize(
    "n_iter, n_chan, n_src, n_fft",
    [
        (50, 2, 2, 4096),
        (20, 6, 2, 4096),
    ],
)
def test_overiva(n_iter, n_chan, n_src, n_fft):

    global mix, ref

    x = mix.clone()
    x = x[:n_chan]

    stft = torchiva.STFT(
        n_fft=n_fft,
    )

    overiva = torchiva.OverIVA_IP(
        n_iter,
        n_src=2,
        model=torchiva.models.GaussModel(),
        proj_back_mic=ref_mic,
        eps=None,
    )

    X = stft(x)

    Y = overiva(X)
    y = stft.inv(Y)

    m = min(ref.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])

    print(f"\nOverIVA  iter:{n_iter:.0f} n_chan:{n_chan:.0f} n_src:{n_src:.0f} SDR", sdr)


    # change source model in forward call
    #Y = overiva(X, model=torchiva.models.NMFModel())
    #y = stft.inv(Y)

    #m = min(ref.shape[-1], y.shape[-1])
    #sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])

    #print(f"\nOverIVA iter:{n_iter:.0f} n_chan:{n_chan:.0f} n_src:{n_src:.0f} SDR", sdr)


@pytest.mark.parametrize(
    "n_iter, n_chan, n_src, n_fft",
    [
        (20, 2, 2, 4096),
    ],
)
def test_ip2(n_iter, n_chan, n_src, n_fft):

    global mix, ref

    x = mix.clone()
    x = x[:n_chan]

    stft = torchiva.STFT(
        n_fft=n_fft,
    )

    ip2 = torchiva.AuxIVA_IP2(
        n_iter,
        model=torchiva.models.GaussModel(),
        proj_back_mic=ref_mic,
        eps=None,
    )

    X = stft(x)
    
    Y = ip2(X, n_iter=n_iter)
    y = stft.inv(Y)

    m = min(ref.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])

    print(f"\nIP2 iter:{n_iter:.0f} n_chan:{n_chan:.0f} n_src:{n_src:.0f} SDR", sdr)


@pytest.mark.parametrize(
    "n_iter, n_chan, n_src, n_fft, n_power_iter",
    [
        (50, 2, 2, 4096, None),
        #(50, 2, 2, 4096, 3),
        (20, 6, 2, 4096, None),
        #(20, 6, 2, 4096, 10),
    ],
)
def test_five(n_iter, n_chan, n_src, n_fft, n_power_iter):

    global mix, ref

    x = mix.clone()
    x = x[:n_chan]

    stft = torchiva.STFT(
        n_fft=n_fft,
    )

    five = torchiva.FIVE(
        n_iter,
        model=torchiva.models.GaussModel(),
        proj_back_mic=ref_mic,
        eps=None,
        n_power_iter=n_power_iter,
    )

    X = stft(x)
    
    Y = five(X)
    y = stft.inv(Y)

    m = min(ref.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])

    print(f"\nFIVE iter:{n_iter:.0f} n_chan:{n_chan:.0f} n_src:{n_src:.0f}", "power_iter", n_power_iter, "SDR", sdr)