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

info = mixinfo["00221"]

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
        #(50, 0, 0, 2, 2, 4096),
        #(20, 0, 0, 6, 2, 4096),
        #(50, 1, 5, 2, 2, 1024),
        (20, 1, 5, 6, 2, 1024),
        
    ],
)
def test_overtiss(n_iter, delay, tap, n_chan, n_src, n_fft):

    global mix, ref

    x = mix.clone()
    x = x[:n_chan]

    stft = torchiva.STFT(
        n_fft=n_fft,
    )

    overtiss1 = torchiva.OverISS_T(
        n_iter,
        n_taps=tap,
        n_delay=delay,
        n_src=2,
        model = torchiva.models.GaussModel(),
        proj_back_mic=ref_mic,
        use_dmc=False,
        eps=None,
    )

    overtiss2 = torchiva.OverISS_T_2(
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

    Y = overtiss1(X)
    y = stft.inv(Y)

    m = min(ref.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])

    print(f"\nOverTISS  iter:{n_iter:.0f} delay:{delay:.0f} tap:{tap:.0f} n_chan:{n_chan:.0f} n_src:{n_src:.0f} SDR", sdr)

    """
    # to see if the result is the same as the ''OverISS_T_2''
    # probably does not work well when n_chan > n_src

    Y = overtiss2(X)
    y = stft.inv(Y)

    m = min(ref.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])

    print(f"OverTISS2 iter:{n_iter:.0f} delay:{delay:.0f} tap:{tap:.0f} n_chan:{n_chan:.0f} n_src:{n_src:.0f} SDR", sdr)
    """
    
    """
    # to see if the result is the same as the normal ''T-ISS''

    tiss = torchiva.AuxIVA_T_ISS(
        model = torchiva.models.NMFModel(),
        n_taps=tap,
        n_delay=delay,
        n_iter=n_iter,
        proj_back=True,
    )

    Y = tiss(X)
    y = stft.inv(Y)

    if n_src < n_chan:
        y = torchiva.select_most_energetic(y, n_src, dim=-2, dim_reduc=-1)
    
    m = min(ref.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])

    # performance differs from OverTISS due to the init param. of NMF source model
    print(f"TISS      iter:{n_iter:.0f} delay:{delay:.0f} tap:{tap:.0f} n_chan:{n_chan:.0f} n_src:{n_src:.0f} SDR", sdr)
    """

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


    """
    # to see if the result is the same as the original functional ''five''

    Y = torchiva.five(
        X,
        n_iter=n_iter, 
        model=torchiva.models.GaussModel(),
        ref_mic=ref_mic,
        use_wiener=False,
        use_n_power_iter=n_power_iter    
    )
    y = stft.inv(Y)

    m = min(ref.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])

    print(f"\nFIVEorg iter:{n_iter:.0f} n_chan:{n_chan:.0f} n_src:{n_src:.0f}", "power_iter", n_power_iter, "SDR", sdr)
    """

def check_all():
    global mixinfo
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    n_iter, tap, delay = 50, 5, 1
    n_chan, n_src = 2, 2
    n_fft = 1024

    overtiss = torchiva.OverISS_T_4(
        n_iter,
        n_taps=tap,
        n_delay=delay,
        n_src=2,
        model = torchiva.models.GaussModel(),
        proj_back_mic=0,
        use_dmc=False,
        eps=1e-5,
    )

    tiss = torchiva.AuxIVA_T_ISS(
        model = torchiva.models.GaussModel(),
        n_taps=tap,
        n_delay=delay,
        n_iter=n_iter,
        proj_back=True,
    )

    stft = torchiva.STFT(
        n_fft=n_fft,
    )
    stft = stft.to(device)

    overtiss_sdr_total = 0
    tiss_sdr_total = 0

    with torch.no_grad():

        for i, (key, info) in enumerate(mixinfo.items()):
            mix, fs = torchaudio.load(Path("wsj1_6ch") / (Path("").joinpath(*Path(info['wav_mixed_noise_reverb']).parts[-4:])))
            ref1, fs = torchaudio.load(Path("wsj1_6ch") / (Path("").joinpath(*Path(info['wav_dpath_image_anechoic'][0]).parts[-4:])))
            ref2, fs = torchaudio.load(Path("wsj1_6ch") / (Path("").joinpath(*Path(info['wav_dpath_image_anechoic'][1]).parts[-4:])))
            ref = torch.stack((ref1[ref_mic], ref2[ref_mic]),dim=0)

            mix = mix[:n_chan]

            mix, ref = mix.to(device), ref.to(device)

            X = stft(mix)

            Y = overtiss(
                X,
                n_src=n_src,
            )
            y = stft.inv(Y)

            m = min(ref.shape[-1], y.shape[-1])
            sdr1, sir1, sar1, perm1 = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])
            overtiss_sdr_total += sdr1.mean()

            Y = tiss(
                X,
            )

            y = stft.inv(Y)
            if n_src < n_chan:
                y = torchiva.most_energetic(y, n_src, dim=-2, dim_reduc=-1)
            
            m = min(ref.shape[-1], y.shape[-1])
            sdr2, sir2, sar2, perm2 = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])
            tiss_sdr_total += sdr2.mean()
            
            if abs(sdr1.mean()-sdr2.mean())>1:
                print(key, "OverTISS: ",sdr1.cpu().numpy(),"TISS: ", sdr2.cpu().numpy())
            #if i==3:
            #    break

            if hasattr(overtiss.model, "reset"):
                overtiss.model.reset()
            if hasattr(tiss.model, "reset"):
                tiss.model.reset()


    print("OVER-TISS:", overtiss_sdr_total / (i+1))
    print("TISS     :", tiss_sdr_total / (i+1))


if __name__=="__main__":
    check_all()