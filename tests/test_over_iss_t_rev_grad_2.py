import argparse
import json
import time
from pathlib import Path

import fast_bss_eval
import numpy as np
import torch
import torch as pt
import torchaudio

# We will first validate the numpy backend
import torchiva as bss

DATA_DIR = Path("bss_speech_dataset/data")
DATA_META = DATA_DIR / "metadata.json"
REF_MIC = 0
RTOL = 1e-5


def make_batch_array(lst):

    m = min([x.shape[-1] for x in lst])
    return pt.cat([x[None, :, :m] for x in lst], dim=0)


def adjust_scale_format_int16(*arrays):
    M = 2**15 / max([a.abs().max() for a in arrays])
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
        torch.mean(bss.linalg.mag_sq(X), dim=(-2, -1), keepdim=True), min=1e-6
    )
    g = torch.sqrt(g)
    X = bss.linalg.divide(X, g)
    return X, g


def unscale(X, g):
    return X * g


def manual_seed_all(seed):
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == "__main__":

    manual_seed_all(0)

    source_models = list(bss.models.source_models.keys())

    parser = argparse.ArgumentParser(description="Separation example")
    parser.add_argument(
        "--no_pb", action="store_true", help="Deactivate projection back"
    )
    parser.add_argument(
        "-r",
        "--rooms",
        default=[0],
        metavar="ROOMS",
        type=int,
        nargs="+",
        help="Room number",
    )
    parser.add_argument("--n_fft", default=256, type=int, help="STFT FFT size")
    parser.add_argument("--hop", type=int, help="STFT hop length size")
    parser.add_argument(
        "--window",
        type=bss.Window,
        choices=bss.window_types,
        help="The STFT window type",
    )
    parser.add_argument(
        "-n", "--n_iter", default=1, type=int, help="Number of iterations"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default="wsj1_2345_db_6m2s/wsj1_2_mix_m6/eval92",
        help="Location of dataset",
    )
    parser.add_argument(
        "--channels",
        "-c",
        type=int,
        help="Number of channels to use",
    )
    parser.add_argument("--snr", default=40, type=float, help="Signal-to-Noise Ratio")
    args = parser.parse_args()

    metafilename = args.dataset / "mixinfo_noise.json"
    with open(metafilename, "r") as f:
        metadata = json.load(f)

    rooms = list(metadata.values())

    assert all(
        [r >= 0 and r < len(rooms) for r in args.rooms]
    ), f"Room must be between 0 and {len(rooms) - 1}"

    t60 = [rooms[r]["rir_info_t60"] for r in args.rooms]
    print(f"Using rooms {args.rooms} with T60={t60}")

    # choose and read the audio files

    mix_lst = []
    ref_lst = []
    for room in args.rooms:

        # the mixtures
        fn_mix = Path(rooms[room]["wav_dpath_mixed_reverberant"])
        fn_mix = Path("").joinpath(*fn_mix.parts[-3:])
        fn_mix = args.dataset / fn_mix
        mix, fs_1 = torchaudio.load(fn_mix)

        mix_lst.append(mix)

        # the reference
        ref_fns_list = rooms[room]["wav_dpath_image_anechoic"]
        ref_fns = [Path(p) for p in ref_fns_list]
        ref_fns = [Path("").joinpath(*fn.parts[-3:]) for fn in ref_fns]

        # now load the references
        audio_ref_list = []
        for fn in ref_fns:
            audio, fs_2 = torchaudio.load(args.dataset / fn)

            assert fs_1 == fs_2

            audio_ref_list.append(audio[[REF_MIC], :])

        ref = pt.cat(audio_ref_list, axis=0)
        ref_lst.append(ref)

    fs = fs_1

    mix = make_batch_array(mix_lst)
    ref = make_batch_array(ref_lst)
    print(mix.shape, ref.shape)

    n_src = ref.shape[-2]
    n_chan = n_src
    mix = mix[..., :n_chan, :]

    print(f"Using {n_chan} channels to separate {n_src} sources.")

    if len(args.rooms) == 1:
        mix = mix[0]
        ref = ref[0]

    # STFT parameters
    if args.hop is None:
        args.hop = args.n_fft // 2
    stft = bss.STFT(args.n_fft, hop_length=args.hop, window=args.window)

    # convert to pytorch tensor if necessary
    print(f"Is GPU available ? {pt.cuda.is_available()}")
    device = pt.device(0 if pt.cuda.is_available() else "cpu")
    mix = mix.to(device)
    ref = ref.to(device)
    stft = stft.to(device)
    print(f"Using device: {device}")

    # STFT
    X = stft(mix)  # copy for back projection (numpy/torch compatible)

    X, g = scale(X)

    X2 = X.clone()

    X.requires_grad_()
    X2.requires_grad_()

    t1 = time.perf_counter()

    n_freq = args.n_fft // 2 + 1
    n_hidden = 128
    dropout_p = 0.1

    # model1 = bss.models.SimpleModel(n_freq=args.n_fft // 2 + 1, n_mels=16)
    model1 = bss.models.GLUMask(
        n_input=n_freq, n_output=n_freq, n_hidden=n_hidden, dropout_p=dropout_p
    )
    model1 = model1.to(device)

    # model2 = bss.models.SimpleModel(n_freq=args.n_fft // 2 + 1, n_mels=16)
    model2 = bss.models.GLUMask(
        n_input=n_freq, n_output=n_freq, n_hidden=n_hidden, dropout_p=dropout_p
    )
    model2 = model2.to(device)

    # make sure both models are initialized the same
    with torch.no_grad():
        for p1, p2 in zip(model1.parameters(), model2.parameters()):
            p1.data[:] = p2.data

    set_requires_grad_(model1)
    set_requires_grad_(model2)

    # Separation normal
    bss_algo_rev = bss.OverISS_T(
        model=model1,
        n_taps=5,
        n_delay=1,
        proj_back=not args.no_pb,
        use_dmc=True,
        eps=1e-3,
    )

    # Separation with backpropagation
    manual_seed_all(0)
    Y1 = bss_algo_rev(X, n_iter=args.n_iter)
    Y1 = unscale(Y1, g)

    # Separation reversible
    manual_seed_all(0)
    Y2 = bss.iss_t_rev(
        X2,
        model2,
        n_iter=args.n_iter,
        n_taps=5,
        n_delay=1,
        eps=1e-3,
        proj_back=not args.no_pb,
    )
    Y2 = unscale(Y2, g)

    def reconstruct_eval(Y):
        y = stft.inv(Y)  # (n_samples, n_channels)
        m = min([ref.shape[-1], y.shape[-1]])
        sdr, *_ = fast_bss_eval.si_sdr(ref[..., :m], y[..., :m])
        return sdr.mean()

    sdr1 = reconstruct_eval(Y1)
    sdr2 = reconstruct_eval(Y2)

    sdr1.backward()
    sdr2.backward()

    grads_1 = [p.grad for p in model1.parameters()]
    grads_2 = [p.grad for p in model2.parameters()]

    print(sdr1)
    print(sdr2)

    if grads_1[0] is not None:
        # print(grads_1[0])
        # print(grads_2[0])
        print(
            torch.norm(grads_1[0]),
            torch.norm(grads_2[0]),
            torch.norm(grads_1[0] - grads_2[0]),
        )
    else:
        print(X2.requires_grad)
        print("yo!")
        print("is X the same tensor as X2?", X is X2)
        print(torch.norm(X.grad - X2.grad))

    # breakpoint()
