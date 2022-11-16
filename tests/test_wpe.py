import torchiva
import fast_bss_eval
import torch
import torchaudio
import json
import pytest
from pathlib import Path

torch.manual_seed(0)
torch.backends.cudnn.deterministic = True

ref_mic = 0

with open(Path("wsj1_6ch") / "dev93" / "mixinfo_noise.json") as f:
    mixinfo = json.load(f)

info = mixinfo["00223"]
dtp = torch.float32

mix, fs = torchaudio.load(
    Path("wsj1_6ch")
    / (Path("").joinpath(*Path(info["wav_mixed_noise_reverb"]).parts[-4:]))
)
mix = mix[[ref_mic]].type(dtp)

ref_reverb, fs = torchaudio.load(
    Path("wsj1_6ch")
    / (Path("").joinpath(*Path(info["wav_dpath_image_reverberant"][0]).parts[-4:]))
)
ref_anechoic, fs = torchaudio.load(
    Path("wsj1_6ch")
    / (Path("").joinpath(*Path(info["wav_dpath_image_anechoic"][0]).parts[-4:]))
)
ref_reverb = ref_reverb[[ref_mic]].type(dtp)
ref_anechoic = ref_anechoic[[ref_mic]].type(dtp)


@pytest.mark.parametrize(
    "n_iter, delay, tap, n_fft",
    [
        (3, 3, 10, 512),
        # (3, 3, 10, 512),
    ],
)
def test_wpe(n_iter, delay, tap, n_fft):

    global ref_reverb, ref_anechoic

    x = ref_reverb.clone()

    stft = torchiva.STFT(
        n_fft=n_fft,
    )

    wpe = torchiva.WPE(
        n_iter=n_iter,
        n_taps=tap,
        n_delay=delay,
        model=None,
        eps=1e-3,
    )

    X = stft(x)
    Y = wpe(X)
    y = stft.inv(Y)

    m = min(ref_anechoic.shape[-1], y.shape[-1])
    sdr = fast_bss_eval.sdr(ref_anechoic[:, :m], y[:, :m])
    sdr_org = fast_bss_eval.sdr(ref_anechoic[:, :m], ref_reverb[:, :m])
    print("reverb: ", sdr_org, "dereverb", sdr)

    # print(f"\nWPE  iter:{n_iter:.0f} delay:{delay:.0f} tap:{tap:.0f} SDR", sdr, "SDRorg", sdr_org)
