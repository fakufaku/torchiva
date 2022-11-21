# Copyright (c) 2022 Robin Scheibler
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import argparse
import json
import warnings
from pathlib import Path

import torch
import torchaudio
import torchiva
import yaml

from urllib.parse import urlparse
from urllib.request import urlretrieve

from .loader import load_separator

DEFAULT_MODEL_URL = (
    "https://raw.githubusercontent.com/fakufaku/torchiva/master/trained_models/tiss"
)


def separate_one_file(separator, path_in, path_out, n_src, n_chan, device):
    mix, fs = torchaudio.load(path_in)

    mix = mix.to(device)

    # limit numer of channel if necessary
    if mix.shape[-2] > n_chan:
        mix = mix[..., :n_chan, :]

    with torch.no_grad():
        y = separator(mix[..., :n_src, :])

        # if args.n_src > n_ref, select most energetic n_ref sources
        if y.shape[-2] > n_src:
            y = torchiva.select_most_energetic(
                y, num=ref.shape[-2], dim=-2, dim_reduc=-1
            )

        # make same size as input
        pad_len = mix.shape[-1] - y.shape[-1]
        y = torch.nn.functional.pad(y, (0, pad_len))

    y = y.cpu()

    torchaudio.save(path_out, y, fs)


def device_type(device):
    if device.isnumeric():
        return f"cuda:{device}"
    else:
        return device


def get_parser():
    algo_choices = [a.value for a in list(torchiva.nn.SepAlgo)]
    parser = argparse.ArgumentParser(description="Separation example")

    # global arguments
    parser.add_argument(
        "input",
        type=Path,
        help="Path to input wav file or folder",
    )
    parser.add_argument("output", type=Path, help="Path to output wav file or folder")
    parser.add_argument(
        "-q", "--quiet", action="store_true", help="Show processed file names"
    )
    parser.add_argument(
        "--device", default="cpu", type=device_type, help="Device to use (default: cpu)"
    )
    parser.add_argument(
        "-m", "--mic", default=2, type=int, help="Maximum number of channels"
    )
    parser.add_argument("-s", "--src", default=2, type=int, help="Number of sources")

    stft_grp = parser.add_argument_group("STFT Parameters")
    stft_grp.add_argument(
        "--nfft", type=int, default=2048, help="Length of the FFT in the STFT"
    )
    stft_grp.add_argument("--hop", type=int, help="Shift length of the STFT")
    stft_grp.add_argument("--win", type=str, help="Window type for the STFT")

    iva_grp = parser.add_argument_group("IVA Parameters")
    iva_grp.add_argument(
        "--algo",
        "-a",
        type=str,
        choices=algo_choices,
        help="IVA spatial update algorithm",
    )
    iva_grp.add_argument(
        "-n", "--n_iter", default=20, type=int, help="Number of iterations"
    )
    iva_grp.add_argument(
        "--taps", "-t", type=int, default=0, help="Number of dereverberation taps"
    )
    iva_grp.add_argument(
        "--delay", "-d", type=int, default=0, help="Delay to insert for dereverberation"
    )
    iva_grp.add_argument("--ref", type=int, default=0, help="Reference channel")
    iva_grp.add_argument(
        "--eps", type=float, help="Small constant to protect division in the algorithm"
    )

    src_grp = parser.add_argument_group("Source model parameters")
    src_grp.add_argument(
        "--model-type",
        type=str,
        default="nn",
        choices=["nn", "laplace", "gauss", "nmf"],
        help="Source model type",
    )
    src_grp.add_argument(
        "--model-path",
        type=str,
        default=DEFAULT_MODEL_URL,
        help="Neural source model parameter file path or URL",
    )
    src_grp.add_argument(
        "--n-basis", type=int, default=2, help="Number of basis functions for NMF"
    )
    src_grp.add_argument(
        "--model-eps", type=float, help="Small constant to protect division"
    )

    return parser


def main(args):

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert args.src <= args.mic

    if args.algo == "five" and args.src > 1:
        args.src = 1
    elif args.algo == "ip2" and not (args.mic == 2 and args.src <= 2):
        raise ValueError(
            "The algorith IP2 can only be used with 2 channels and at most 2 sources"
        )

    # load the pre-trained model
    if args.model_type == "nn":
        kwargs = dict(
            n_iter=args.n_iter,
            n_src=args.src,
            proj_back_mic=args.ref,
            use_dmc=False,
        )

        if args.algo is not None:
            kwargs["algo"] = args.algo

        separator = load_separator(args.model_path, **kwargs)

    else:
        if args.algo is None:
            args.algo = "tiss"
        elif args.algo in ["mvdr", "mwf", "gev"]:
            raise ValueError(
                "Beamforming separation models require a pre-trained DNN model"
            )

        if args.model_type == "laplace":
            source_model = torchiva.models.LaplaceModel(eps=args.model_eps)
        elif args.model_type == "gauss":
            source_model = torchiva.models.GaussModel(eps=args.model_eps)
        elif args.model_type == "nmf":
            source_model = torchiva.models.NMFModel(
                n_basis=args.n_basis, eps=args.model_eps
            )
        else:
            raise ValueError(f"Unknown model type {args.model_type}")

        separator = torchiva.nn.BSSSeparator(
            n_fft=args.nfft,
            hop_length=args.hop,
            window=args.win,
            n_taps=args.taps,
            n_delay=args.delay,
            n_src=args.src,
            algo=args.algo,
            source_model=source_model,
            proj_back_mic=args.ref,
            use_dmc=False,
            eps=args.eps,
        )

    if args.device is not None:
        separator = separator.to(args.device)

    if args.input.is_dir():
        args.output.mkdir(exist_ok=True, parents=True)
        if not args.output.is_dir():
            raise ValueError(f"{args.output} is not a folder")
        if args.output == args.input:
            raise ValueError("Input and output should be different folders")

        for path_in in args.input.rglob("*.wav"):
            path_out = args.output / path_in.name
            separate_one_file(
                separator, path_in, path_out, args.src, args.mic, args.device
            )
            if not args.quiet:
                print(f"{path_in} -> {path_out}")

    else:

        if args.input.suffix != ".wav":
            raise ValueError("The input file should be a wav file")

        if args.output.is_dir():
            path_out = args.output / args.input.name
        else:
            path_out = args.output

        if args.output == args.input:
            raise ValueError("Input and output should be different files")

        separate_one_file(
            separator, args.input, path_out, args.src, args.mic, args.device
        )
        if not args.quiet:
            print(f"{args.input} -> {path_out}")


if __name__ == "__main__":
    torch.manual_seed(0)
    parser = get_parser()
    args = parser.parse_args()
    main(args)
