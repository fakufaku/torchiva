import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch as pt
import torch
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
        torch.mean(bss.linalg.mag_sq(X), dim=(-2, -1), keepdim=True), min=1e-6
    )
    g = torch.sqrt(g)
    X = bss.linalg.divide(X, g)
    return X, g


def unscale(X, g):
    return X * g


if __name__ == "__main__":

    np.random.seed(0)
    torch.manual_seed(0)

    source_models = list(bss.models.source_models.keys())

    parser = argparse.ArgumentParser(description="Separation example")
    parser.add_argument(
        "--no_pb", action="store_true", help="Deactivate projection back"
    )
    parser.add_argument("--rev", action="store_true", help="Use reversible algorithm")
    parser.add_argument(
        "-p",
        type=float,
        help="Outer norm",
    )
    parser.add_argument(
        "-q",
        type=float,
        help="Inner norm",
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
        "-n", "--n_iter", default=10, type=int, help="Number of iterations"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default="wsj1_2345_db/wsj1_2_mix_m2/eval92",
        help="Location of dataset",
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
        fn_mix = Path("").joinpath(*fn_mix.parts[-2:])
        fn_mix = args.dataset / fn_mix
        mix, fs_1 = torchaudio.load(fn_mix)

        mix_lst.append(mix)

        # the reference
        ref_fns_list = rooms[room]["wav_dpath_image_anechoic"]
        ref_fns = [Path(p) for p in ref_fns_list]
        ref_fns = [Path("").joinpath(*fn.parts[-2:]) for fn in ref_fns]

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

    if len(args.rooms) == 1:
        mix = mix[0]
        ref = ref[0]

    # STFT parameters
    if args.hop is None:
        args.hop = args.n_fft // 2
    stft = bss.STFT(args.n_fft, hop_length=args.hop, window=args.window)

    # convert to pytorch tensor if necessary
    print(f"Is GPU available ? {pt.cuda.is_available()}")
    device = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")
    mix = mix.to(device)
    ref = ref.to(device)

    # STFT
    X = stft(mix)  # copy for back projection (numpy/torch compatible)

    X, g = scale(X)

    X2 = X.clone()

    X.requires_grad_()
    X2.requires_grad_()

    t1 = time.perf_counter()

    model = bss.models.SimpleModel(n_freq=args.n_fft // 2 + 1, n_mels=16)
    model = model.to(device)

    bss_algo = bss.AuxIVA_T_ISS(
        model=model, n_taps=5, n_delay=1, proj_back=not args.no_pb
    )

    def reconstruct_eval(Y):
        y = stft.inv(Y)  # (n_samples, n_channels)
        m = min([ref.shape[-1], y.shape[-1]])
        sdr, perm = bss.metrics.si_sdr(ref[..., :m], y[..., :m])
        return sdr.mean()

    ### optimization ###
    optim_epoch = 50
    lr = 0.01
    mom = 0.5
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=mom)
    method = "reversible"
    method = "regular"

    for epoch in range(optim_epoch):
        optimizer.zero_grad()

        if args.rev:
            # Separation reversible
            Y = bss.iss_t_rev(
                X,
                model,
                n_iter=args.n_iter,
                n_taps=5,
                n_delay=1,
                proj_back=not args.no_pb,
            )

        else:
            Y = bss_algo(X, n_iter=args.n_iter)

        Y = unscale(Y, g)

        neg_sdr = -reconstruct_eval(Y)

        print(f"{epoch} SDR={-neg_sdr:.2f} dB")

        neg_sdr.backward()
        optimizer.step()
