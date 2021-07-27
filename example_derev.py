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


if __name__ == "__main__":

    np.random.seed(0)

    source_models = list(bss.source_models.keys())

    parser = argparse.ArgumentParser(description="Separation example")
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
        "-d",
        "--source_model",
        default=source_models[0],
        choices=source_models,
        type=str,
        help="Source model",
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

    t1 = time.perf_counter()

    # Separation
    bss_algo = bss.AuxIVA_T_ISS(
        model=bss.source_models[args.source_model], n_taps=5, n_delay=1
    )

    Y = bss_algo(X, n_iter=args.n_iter)

    t2 = time.perf_counter()

    print(f"Separation time: {t2 - t1:.3f} s")

    # Projection back

    if args.p is not None:
        Y = bss.minimum_distortion(Y, X[..., REF_MIC, :, :], p=args.p, q=args.q)
    else:
        Y = bss.projection_back(Y, X[..., REF_MIC, :, :])

    t3 = time.perf_counter()

    print(f"Proj. back time: {t3 - t2:.3f} s")

    # iSTFT
    y = stft.inv(Y)  # (n_samples, n_channels)

    t4 = time.perf_counter()

    # Evaluate
    m = min([ref.shape[-1], y.shape[-1]])

    # scale invaliant metric
    sdr, sir, sar, perm = bss.metrics.si_bss_eval(ref[..., :m], y[..., :m])

    t5 = time.perf_counter()

    print(f"Eval. back time: {t5 - t4:.3f} s")

    mix = mix.cpu()
    ref = ref.cpu()
    y = y.cpu()

    mix, ref, y = adjust_scale_format_int16(mix, ref, y)

    if mix.ndim == 2:
        torchaudio.save("example_mix.wav", mix, fs)
        torchaudio.save("example_ref.wav", ref[..., :m], fs)
        torchaudio.save("example_output.wav", y[..., :m], fs)

    # Reorder the signals
    print("SDR:", sdr)
    print("SIR:", sir)
