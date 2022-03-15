import argparse
import json
import time
from pathlib import Path

import fast_bss_eval
import numpy as np
import torch
from torch.nn.parallel import data_parallel
import torchaudio

# We will first validate the numpy backend
import torchiva as bss

DATA_DIR = Path("bss_speech_dataset/data")
DATA_META = DATA_DIR / "metadata.json"
REF_MIC = 0
RTOL = 1e-5

def scale(X):
    g = torch.clamp(
        torch.mean(bss.linalg.mag_sq(X), dim=(-2, -1), keepdim=True), min=1e-6
    )
    g = torch.sqrt(g)
    X = bss.linalg.divide(X, g)
    return X, g


def unscale(X, g):
    return X * g

class Separation(torch.nn.Module):
    def __init__(self, n_fft, hop, window, source_model, n_iter, no_pb=False, use_dmc=True):
        super().__init__()
        self.stft = bss.STFT(args.n_fft, hop_length=args.hop, window=args.window)
        self.source_model = source_model
        self.n_iter = n_iter
        self.iss_t = bss.OverISS_T(
            model=source_model,
            n_taps=5,
            n_delay=1,
            proj_back=not no_pb,
            use_dmc=use_dmc,
            eps=1e-3,
        )

    def forward(self, x, ref):
        X = self.stft(x)

        X, g = scale(X)

        Y = self.iss_t(X, n_iter=self.n_iter)
    
        Y = unscale(Y, g)

        y = self.stft.inv(Y)  # (n_samples, n_channels)

        m = min([ref.shape[-1], y.shape[-1]])
        sdr = fast_bss_eval.si_sdr(ref[..., :m], y[..., :m])

        return sdr.sum()


def make_batch_array(lst):

    m = max([x.shape[-1] for x in lst])
    out = torch.zeros((len(lst), lst[0].shape[0], m))
    for i, sig in enumerate(lst):
        out[i, :, :sig.shape[1]] = sig
    return out


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
        "--channels", "-c", type=int, help="Number of channels to use",
    )
    parser.add_argument("--snr", default=40, type=float, help="Signal-to-Noise Ratio")
    parser.add_argument("--use_dp", action="store_true", help="Use DataParallel")
    parser.add_argument("--use_dmc", action="store_true", help="Use checkpointing")
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

        ref = torch.cat(audio_ref_list, axis=0)
        ref_lst.append(ref)

    fs = fs_1

    mix = make_batch_array(mix_lst)
    ref = make_batch_array(ref_lst)
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
            "The number of channels should be more than "
            f"the number of sources (i.e., {n_src})"
        )

    print(f"Using {n_chan} channels to separate {n_src} sources.")

    if len(args.rooms) == 1:
        mix = mix[0]
        ref = ref[0]

    # STFT parameters
    if args.hop is None:
        args.hop = args.n_fft // 2

    # convert to pytorch tensor if necessary
    print(f"Is GPU available ? {torch.cuda.is_available()}")
    device = torch.device(0 if torch.cuda.is_available() else "cpu")
    mix = mix.to(device)
    ref = ref.to(device)
    print(f"Using device: {device}")

    n_freq = args.n_fft // 2 + 1
    n_hidden = 128
    dropout_p = 0.0

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
    bss_algo_1 = Separation(args.n_fft, args.hop, args.window, model1, args.n_iter, args.no_pb, args.use_dmc).to(device)
    bss_algo_2 = Separation(args.n_fft, args.hop, args.window, model2, args.n_iter, args.no_pb, args.use_dmc).to(device)

    # Separation with backpropagation
    manual_seed_all(0)
    sdr1 = bss_algo_1(mix, ref)

    # Separation reversible
    manual_seed_all(0)
    if args.use_dp:
        print("Use DP!")
        sdr2 = data_parallel(
            bss_algo_2, inputs=(mix, ref), device_ids=(0, 1)
        )
        sdr2 = sdr2.sum()
    else:
        sdr2 = bss_algo_2(mix, ref)

    sdr1.backward()
    sdr2.backward()

    grads_1 = [p.grad for p in model1.parameters()]
    grads_2 = [p.grad for p in model2.parameters()]

    print(sdr1, sdr2)

    if grads_1[0] is not None:
        print(
            torch.norm(grads_1[0]),
            torch.norm(grads_2[0]),
            torch.norm(grads_1[0] - grads_2[0]),
        )
