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
import torchiva

DATA_DIR = Path("bss_speech_dataset/data")
DATA_META = DATA_DIR / "metadata.json"
REF_MIC = 0
RTOL = 1e-5


def make_batch_array(lst, adjust="min"):
    if adjust == "max":
        m = max([x.shape[-1] for x in lst])
        batch = lst[0].new_zeros((len(lst), lst[0].shape[0], m))
        for i, example in enumerate(lst):
            batch[i, :, : example.shape[1]] = example
        return batch
    elif adjust == "min":
        m = min([x.shape[-1] for x in lst])
        return pt.cat([x[None, :, :m] for x in lst], dim=0)
    else:
        raise NotImplementedError()


def adjust_scale_format_int16(*arrays):
    M = 2 ** 15 / max([a.abs().max() for a in arrays])
    out_arrays = []
    for a in arrays:
        out_arrays.append((a * M).type(torch.int16))
    return out_arrays


if __name__ == "__main__":

    np.random.seed(0)

    source_models = list(torchiva.models.source_models.keys())

    parser = argparse.ArgumentParser(description="Separation example")
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
        type=str,
        choices=["hann", "hamming"],
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
        default="wsj1_2345_db_6m2s/wsj1_2_mix_m6/eval92",
        help="Location of dataset",
    )
    parser.add_argument(
        "--channels",
        "-c",
        type=int,
        help="Number of channels to use",
    )
    parser.add_argument(
        "--ref_reverb", action="store_true", help="Use reverberant signal as reference"
    )
    parser.add_argument("--snr", default=40, type=float, help="Signal-to-Noise Ratio")
    parser.add_argument(
        "--v2", action="store_true", help="Use version 2 (matrix inverse-free)."
    )
    parser.add_argument(
        "--taps", "-t", type=int, default=0, help="Number of dereverberation taps."
    )
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
        if args.ref_reverb:
            ref_fns_list = rooms[room]["wav_dpath_image_reverberant"]
        else:
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

    mix = make_batch_array(mix_lst, adjust="max")
    ref = make_batch_array(ref_lst, adjust="max")
    print(mix.shape, ref.shape)

    n_src = ref.shape[-2]

    if args.channels is None:
        n_chan = mix.shape[-2]
    elif args.channels >= n_src:
        n_chan = args.channels
        if n_chan < mix.shape[-2]:
            mix = mix[..., :n_chan, :]
    else:
        raise ValueError(
            f"The number of channels should be more than "
            f"the number of sources (i.e., {n_src})"
        )

    print(f"Using {n_chan} channels to separate {n_src} sources.")

    if len(args.rooms) == 1:
        mix = mix[0]
        ref = ref[0]

    # STFT parameters
    if args.hop is None:
        args.hop = args.n_fft // 2
        n_delay = 1
    else:
        n_delay = args.n_fft // args.hop - 1

    stft = torchiva.STFT(args.n_fft, hop_length=args.hop, window=args.window)

    # convert to pytorch tensor if necessary
    print(f"Is GPU available ? {pt.cuda.is_available()}")
    device = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")
    mix = mix.to(device)
    ref = ref.to(device)
    stft = stft.to(device)

    # infer number of sources from reference file
    n_src = ref.shape[1]

    # STFT
    X = stft(mix)  # copy for back projection (numpy/torch compatible)

    t1 = time.perf_counter()

    model = torchiva.models.source_models[args.source_model]
    """
    model=torchiva.models.GLUMask(
    n_input=args.n_fft // 2 + 1, n_output=1, n_hidden=128
    )
    .to(device)
    .requires_grad_(),
    """
    use_dmc = True

    # Separation
    if args.v2:
        bss_algo = torchiva.OverISS_T_2(
            model=model,
            n_taps=args.taps,
            n_delay=n_delay,
            proj_back=True,
            verbose=True,
            eps=1e-15,
            use_dmc=use_dmc,
        )
    else:
        bss_algo = torchiva.OverISS_T(
            model=model,
            n_taps=args.taps,
            n_delay=n_delay,
            proj_back=True,
            verbose=True,
            eps=1e-15,
            use_dmc=use_dmc,
        )

    Y = bss_algo(X, n_iter=args.n_iter, n_src=2)

    t2 = time.perf_counter()

    print(f"Separation time: {t2 - t1:.3f} s")

    # iSTFT
    y = stft.inv(Y)  # (n_samples, n_channels)

    t4 = time.perf_counter()

    # Evaluate
    m = min([ref.shape[-1], y.shape[-1]])

    # scale invaliant metric
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[..., :m], y[..., :m])

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
    # print("SIR:", sir)
    print("SDR (mean):", sdr.mean())
    print("SIR (mean):", sir.mean())
