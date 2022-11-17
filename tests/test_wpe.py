import torchiva
import fast_bss_eval
import torch
import torchaudio
import json
import pytest
from pathlib import Path

from examples.samples.read_samples import read_samples

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

mix, ref_reverb, ref_anechoic = read_samples(ref_mic=0)

@pytest.mark.parametrize(
    "n_iter, delay, tap, n_fft, tol_db", [(10, 3, 15, 256, 5),],
)
def test_wpe(n_iter, delay, tap, n_fft, tol_db):

    global ref_reverb, ref_anechoic

    x = ref_reverb.clone()

    stft = torchiva.STFT(n_fft=n_fft,)

    wpe = torchiva.WPE(n_iter=n_iter, n_taps=tap, n_delay=delay, model=None, eps=1e-3,)

    X = stft(x)
    Y = wpe(X)
    y = stft.inv(Y)

    m = min(ref_anechoic.shape[-1], y.shape[-1])
    sdr = fast_bss_eval.sdr(ref_anechoic[:, :m], y[:, :m])
    sdr_org = fast_bss_eval.sdr(ref_anechoic[:, :m], ref_reverb[:, :m])

    improv = sdr.mean() - sdr_org.mean()
    print(
        f"\nWPE  iter:{n_iter:.0f} delay:{delay:.0f} tap:{tap:.0f} SDR",
        sdr,
        "SDR original",
        sdr_org,
        "Improv:",
        improv,
    )
    print
    assert improv > tol_db


if __name__ == "__main__":
    test_wpe(10, 3, 15, 256, 0)
