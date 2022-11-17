import torchiva
import fast_bss_eval
import torch
import torchaudio
import json
import pytest
from pathlib import Path
import warnings

from examples.samples.read_samples import read_samples

warnings.simplefilter("ignore")

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

ref_mic = 1

mix, ref_wet, ref_dry = read_samples(ref_mic=ref_mic)


@pytest.mark.parametrize(
    "n_iter, delay, tap, n_chan, n_src, n_fft, target_db",
    [
        (50, 0, 0, 2, 2, 4096, 9),
        (20, 0, 0, 3, 2, 4096, 9),
        (50, 3, 2, 2, 2, 2048, 10),
        (20, 3, 2, 3, 2, 2048, 10),
    ],
)
def test_tiss(n_iter, delay, tap, n_chan, n_src, n_fft, target_db):

    global mix, ref_wet, ref_dry

    if tap > 0:
        ref_loc = ref_dry
    else:
        ref_loc = ref_wet

    ref_loc = ref_dry

    x = mix.clone()
    x = x[:n_chan]

    stft = torchiva.STFT(n_fft=n_fft, hop_length=n_fft // 4)

    overtiss = torchiva.T_ISS(
        n_iter,
        n_taps=tap,
        n_delay=delay,
        n_src=2,
        model=torchiva.models.NMFModel(),
        proj_back_mic=ref_mic,
        use_dmc=False,
        eps=1e-3,
    )

    X = stft(x)

    Y = overtiss(X)
    y = stft.inv(Y)

    m = min(ref_loc.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref_loc[:, :m], y[:, :m])

    print(
        f"\nTISS  iter:{n_iter:.0f} delay:{delay:.0f} tap:{tap:.0f} n_chan:{n_chan:.0f} n_src:{n_src:.0f} SDR",
        sdr,
    )

    assert sdr.mean() > target_db


@pytest.mark.parametrize(
    "n_iter, n_chan, n_src, n_fft, target_db",
    [
        (50, 2, 2, 2048, 4),
        (20, 3, 2, 2048, 4),
    ],
)
def test_iva(n_iter, n_chan, n_src, n_fft, target_db):

    global mix, ref_wet

    x = mix.clone()
    x = x[:n_chan]

    stft = torchiva.STFT(
        n_fft=n_fft,
        hop_length=n_fft // 4,
    )

    overiva = torchiva.AuxIVA_IP(
        n_iter,
        n_src=2,
        model=torchiva.models.LaplaceModel(eps=1e-10),
        proj_back_mic=ref_mic,
        eps=1e-6,
    )

    X = stft(x)

    Y = overiva(X, verbose=False)
    y = stft.inv(Y)

    m = min(ref_wet.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref_wet[:, :m], y[:, :m])

    print(
        f"\nOverIVA  iter:{n_iter:.0f} n_chan:{n_chan:.0f} n_src:{n_src:.0f}", end=" "
    )
    print("SDR", sdr, "SIR", sir)

    assert sdr.mean() > target_db


@pytest.mark.parametrize(
    "n_iter, n_chan, n_src, n_fft, target_db",
    [
        (50, 2, 2, 4096, 12),
    ],
)
def test_ip2(n_iter, n_chan, n_src, n_fft, target_db):

    global mix, ref_wet

    x = mix.clone()
    x = x[:, :n_chan, :]

    stft = torchiva.STFT(
        n_fft=n_fft,
        hop_length=n_fft // 4,
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

    m = min(ref_wet.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref_wet[:, :m], y[:, :m])

    print(f"\nIP2 iter:{n_iter:.0f} n_chan:{n_chan:.0f} n_src:{n_src:.0f} SDR", sdr)


@pytest.mark.parametrize(
    "n_iter, n_chan, n_src, n_fft, n_power_iter, target_db",
    [
        (1, 2, 2, 4096, None, 0),
        (1, 3, 2, 4096, None, 0),
    ],
)
def test_five(n_iter, n_chan, n_src, n_fft, n_power_iter, target_db):

    global mix, ref_wet

    x = mix.clone()
    x = x[:n_chan]

    stft = torchiva.STFT(
        n_fft=n_fft,
        hop_length=n_fft // 4,
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

    m = min(ref_wet.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref_wet[:, :m], y[:, :m])

    print(
        f"\nFIVE iter:{n_iter:.0f} n_chan:{n_chan:.0f} n_src:{n_src:.0f}",
        "power_iter",
        n_power_iter,
        "SDR",
        sdr,
    )

    assert sdr.mean() > target_db


def check_all():
    test_tiss(50, 0, 0, 2, 2, 4096, 0)
    test_tiss(20, 0, 0, 3, 2, 4096, 0)
    test_tiss(50, 3, 2, 2, 2, 2048, 0),
    test_tiss(20, 3, 2, 3, 2, 2048, 0),
    test_iva(50, 2, 2, 2048, 0)
    test_iva(20, 3, 2, 2048, 0)
    test_ip2(20, 2, 2, 4096, 0)
    test_five(1, 2, 2, 4096, None, 0)
    test_five(1, 3, 2, 4096, None, 0)


if __name__ == "__main__":
    check_all()
