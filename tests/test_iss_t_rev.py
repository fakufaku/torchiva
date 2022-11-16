import argparse
import json
import time
from pathlib import Path

import fast_bss_eval
import numpy as np
import pytest
import torch
import torchaudio

# We will first validate the numpy backend
import torchiva
import yaml

REF_MIC = 0
RTOL = 1e-5

import torch
from examples.samples.read_samples import read_samples
from torch import nn
from torchaudio.transforms import MelScale


def set_requires_grad_(module):
    for p in module.parameters():
        p.requires_grad_()


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

    mix, ref = read_samples(ref_mic=REF_MIC)

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
        model=model1, n_taps=5, n_delay=1, proj_back_mic=REF_MIC, eps=1e-3
    )
    Y1 = bss_algo(X, n_iter=n_iter)
    Y1 = unscale(Y1, g)

    # Separation reversible
    bss_algo_rev = torchiva.T_ISS(
        model=model2, n_taps=5, n_delay=1, proj_back_mic=REF_MIC, eps=1e-3, use_dmc=True
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

    err = [
        abs(g1 - g2).max() for g1, g2 in zip(model1.parameters(), model2.parameters())
    ]
    for e in err:
        assert e < 1e-5


@pytest.mark.parametrize(
    "use_dmc, n_fft, optim_epoch, n_iter",
    [(True, 256, 3, 3), (False, 256, 3, 3), (True, 512, 3, 3), (False, 512, 3, 3),],
)
def test_iss_t_rev_opt(use_dmc, n_fft, optim_epoch, n_iter, lr=0.001, device="cpu"):

    torch.manual_seed(0)

    mix, ref = read_samples(ref_mic=REF_MIC)

    stft = torchiva.STFT(n_fft)

    mix = mix.to(device)
    ref = ref.to(device)
    stft = stft.to(device)

    # STFT
    X = stft(mix)  # copy for back projection (numpy/torch compatible)

    X, g = scale(X)

    model = torchiva.models.SimpleModel(n_freq=n_fft // 2 + 1, n_mels=16)
    model = model.to(device)

    eps = 1e-6
    bss_algo = torchiva.T_ISS(
        model=model,
        n_taps=5,
        n_delay=1,
        proj_back_mic=REF_MIC,
        eps=eps,
        use_dmc=use_dmc,
    )

    def reconstruct_eval(Y):
        y = stft.inv(Y)  # (n_samples, n_channels)
        m = min([ref.shape[-1], y.shape[-1]])
        sdr = fast_bss_eval.si_sdr(ref[..., :m], y[..., :m], clamp_db=40)
        return sdr.mean()

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Number of trainable parameters: {pytorch_total_params}")

    sdr_0 = fast_bss_eval.si_sdr(ref, mix).mean()

    for epoch in range(optim_epoch):
        optimizer.zero_grad()

        Y = bss_algo(X, n_iter=n_iter)

        Y = unscale(Y, g)

        neg_sdr = -reconstruct_eval(Y)

        if epoch == 0:
            sdr_epoch_0 = -neg_sdr.clone().detach()

        print(f"{epoch} SDR={-neg_sdr:.2f} dB")

        neg_sdr.backward()
        optimizer.step()

    print(f"{epoch} SDR={-neg_sdr:.2f} dB")

    sdri = -neg_sdr - sdr_epoch_0
    print("Improvement:", sdri)
    assert sdri > 0.75


if __name__ == "__main__":
    test_iss_t_rev_opt(True, 512, 3, 10, device=0)
