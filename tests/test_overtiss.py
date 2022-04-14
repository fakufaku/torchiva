import torchiva
import fast_bss_eval
import torch
import torchaudio
import json
import pytest
from pathlib import Path
import pyroomacoustics as pra 
import numpy as np
import warnings

warnings.simplefilter('ignore')

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

ref_mic=1

with open(Path("wsj1_6ch") / "dev93" / "mixinfo_noise.json") as f:
    mixinfo = json.load(f)

info = mixinfo["00222"]
dtp = torch.float64

#mix, fs = torchaudio.load(Path("wsj1_6ch") / (Path("").joinpath(*Path(info['wav_mixed_noise_reverb']).parts[-4:])))
mix, fs = torchaudio.load(Path("wsj1_6ch") / (Path("").joinpath(*Path(info['wav_dpath_mixed_reverberant']).parts[-4:])))
mix = mix.type(dtp)

#if delay==0 and tap==0:
#    ref1, fs = torchaudio.load(Path("wsj1_6ch") / (Path("").joinpath(*Path(info['wav_dpath_image_reverberant'][0]).parts[-4:])))
#    ref2, fs = torchaudio.load(Path("wsj1_6ch") / (Path("").joinpath(*Path(info['wav_dpath_image_reverberant'][1]).parts[-4:])))
#else:
#    ref1, fs = torchaudio.load(Path("wsj1_6ch") / (Path("").joinpath(*Path(info['wav_dpath_image_anechoic'][0]).parts[-4:])))
#    ref2, fs = torchaudio.load(Path("wsj1_6ch") / (Path("").joinpath(*Path(info['wav_dpath_image_anechoic'][1]).parts[-4:])))

ref1, fs = torchaudio.load(Path("wsj1_6ch") / (Path("").joinpath(*Path(info['wav_dpath_image_anechoic'][0]).parts[-4:])))
ref2, fs = torchaudio.load(Path("wsj1_6ch") / (Path("").joinpath(*Path(info['wav_dpath_image_anechoic'][1]).parts[-4:])))
ref1 = ref1.type(dtp)
ref2 = ref2.type(dtp)
ref = torch.stack((ref1[ref_mic], ref2[ref_mic]),dim=0)


@pytest.mark.parametrize(
    "n_iter, delay, tap, n_chan, n_src, n_fft",
    [
        (20, 0, 0, 2, 2, 4096),
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
        model = torchiva.models.LaplaceModel(),
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
        (20, 2, 2, 4096),
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
        model=torchiva.models.LaplaceModel(),
        proj_back_mic=ref_mic,
        eps=None,
    )

    X = stft(x)

    Y = overiva(X)
    y = stft.inv(Y)

    m = min(ref.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])

    print(f"\nOverIVA  iter:{n_iter:.0f} n_chan:{n_chan:.0f} n_src:{n_src:.0f} SDR", sdr)


    #Y = pra.bss.auxiva(
    #    np.transpose(X.cpu().numpy(), [2,1,0]),
    #    n_src=2,
    #    n_iter=n_iter,
    #    proj_back=True,
    #    model="laplace",
    #)
    #Y = torch.from_numpy(Y).transpose(-1, -3).type_as(X)
    #y = stft.inv(Y)

    #m = min(ref.shape[-1], y.shape[-1])
    #sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])

    #print(f"\nOverIVA_pra  iter:{n_iter:.0f} n_chan:{n_chan:.0f} n_src:{n_src:.0f} SDR", sdr)

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
        model=torchiva.models.LaplaceModel(),
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
        #(0, 2, 2, 4096, None),
        #(0, 6, 2, 4096, None),
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
        model=torchiva.models.LaplaceModel(),
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