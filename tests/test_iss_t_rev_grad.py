import yaml
import argparse
import json
import time
from pathlib import Path

import fast_bss_eval
import numpy as np
import torch
import torchaudio

# We will first validate the numpy backend
import torchiva

REF_MIC = 0
RTOL = 1e-5

import torch
from torch import nn

from torchaudio.transforms import MelScale




def make_batch_array(lst):

    m = max([x.shape[-1] for x in lst])
    return torch.cat([x[None, :, :m] for x in lst], dim=0)


def adjust_scale_format_int16(*arrays):
    M = 2 ** 15 / max([a.abs().max() for a in arrays])
    out_arrays = []
    for a in arrays:
        out_arrays.append((a * M).type(torch.int16))
    return out_arrays


def set_requires_grad_(module):
    for p in module.parameters():
        p.requires_grad_()


def print_params(module):
    for p in module.parameters():
        print(p)


def scale(X):
    g = torch.clamp(
        torch.mean(torchiva.linalg.mag_sq(X), dim=(-2, -1), keepdim=True), min=1e-6
    )
    g = torch.sqrt(g)
    X = torchiva.linalg.divide(X, g)
    return X, g


def unscale(X, g):
    return X * g


def test_iss_t_rev_grad():

    torch.manual_seed(0)

    n_iter = 3
    n_fft = 256
    device = "cpu"


    # choose and read the audio files
    samples_dir = Path(__file__).parent / "../examples/samples"
    with open(samples_dir / "samples_list.yaml", "r") as f:
        samples = yaml.safe_load(f)


    mix_lst = []
    ref_lst = []
    for sample in samples:

        # the mixtures
        mix, fs_1 = torchaudio.load(samples_dir / sample["mix"])
        mix_lst.append(mix)

        # now load the references
        audio_ref_list = []
        for fn in sample["ref"]:
            audio, fs_2 = torchaudio.load(samples_dir / fn)
            assert fs_1 == fs_2
            audio_ref_list.append(audio[[REF_MIC], :])

        ref = torch.cat(audio_ref_list, dim=0)
        ref_lst.append(ref)

    fs = fs_1

    mix = make_batch_array(mix_lst)
    ref = make_batch_array(ref_lst)

    # STFT parameters
    stft = torchiva.STFT(n_fft)

    # transfer to device
    mix = mix.to(device)
    ref = ref.to(device)
    stft = stft.to(device)

    # STFT
    X = stft(mix)  # copy for back projection (numpy/torch compatible)

    X, g = scale(X)

    X2 = X.clone()

    X.requires_grad_()
    X2.requires_grad_()

    t1 = time.perf_counter()

    model1 = torchiva.models.SimpleModel(n_freq=n_fft // 2 + 1, n_mels=16)
    model1 = model1.to(device)

    model2 = torchiva.models.SimpleModel(n_freq=n_fft // 2 + 1, n_mels=16)
    model2 = model2.to(device)

    # make sure both models are initialized the same
    with torch.no_grad():
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p1.data[:] = p2.data

    set_requires_grad_(model1)
    set_requires_grad_(model2)

    # Separation normal
    bss_algo = torchiva.T_ISS(
        model=model1, n_taps=5, n_delay=1, proj_back_mic=0, eps=1e-3
    )
    Y1 = bss_algo(X, n_iter=n_iter)
    Y1 = unscale(Y1, g)

    # Separation reversible
    bss_algo_rev = torchiva.T_ISS(
        model=model2, n_taps=5, n_delay=1, proj_back_mic=0, eps=1e-3, use_dmc=True
    )
    Y2 = bss_algo_rev(X2, n_iter=n_iter)
    Y2 = unscale(Y2, g)

    def reconstruct_eval(Y):
        y = stft.inv(Y)  # (n_samples, n_channels)
        m = min([ref.shape[-1], y.shape[-1]])
        sdr = fast_bss_eval.si_sdr(ref[..., :m], y[..., :m], clamp_db=40)
        return sdr.mean()

    sdr1 = reconstruct_eval(Y1)
    sdr2 = reconstruct_eval(Y2)

    sdr1.backward()
    sdr2.backward()

    assert abs(sdr1 - sdr2) < 1e-4

    err = [abs(g1 - g2).max() for g1, g2 in zip(model1.parameters(), model2.parameters())]
    for e in err:
        assert e < 1e-5
