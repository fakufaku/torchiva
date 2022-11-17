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

DEFAULT_MODEL = "https://raw.githubusercontent.com/fakufaku/torchiva/master/trained_models/tiss"

def separate_one_file(separator, path_in, path_out, n_src, n_chan):
    mix, fs = torchaudio.load(path_in)

    # limit numer of channel if necessary
    if mix.shape[-2] > n_chan:
        mix = mix[..., :n_chan, :]

    with torch.no_grad():
        y = separator(mix[..., : args.src, :])

        # if args.n_src > n_ref, select most energetic n_ref sources
        if y.shape[-2] > n_src:
            y = torchiva.select_most_energetic(
                y, num=ref.shape[-2], dim=-2, dim_reduc=-1
            )

        # make same size as input
        pad_len = mix.shape[-1] - y.shape[-1]
        y = torch.nn.functional.pad(y, (0, pad_len))

    torchaudio.save(path_out, y, fs)


if __name__ == "__main__":

    torch.manual_seed(0)

    parser = argparse.ArgumentParser(description="Separation example")

    parser.add_argument(
        "input", type=Path, help="Path to input wav file or folder",
    )
    parser.add_argument("output", type=Path, help="Path to output wav file or folder")
    parser.add_argument(
        "--model", type=str, default=DEFAULT_MODEL, help="Model parameter path",
    )
    parser.add_argument(
        "-n", "--n_iter", default=20, type=int, help="Number of iterations"
    )
    parser.add_argument(
        "-m", "--mic", default=2, type=int, help="Maximum number of channels"
    )
    parser.add_argument("-s", "--src", default=2, type=int, help="Number of sources")

    args = parser.parse_args()

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    assert args.src <= args.mic

    # load the pre-trained model
    separator = load_separator(args.model, n_iter=args.n_iter, n_src=args.src,)

    if args.input.is_dir():
        args.output.mkdir(exist_ok=True, parents=True)
        if not args.output.is_dir():
            raise ValueError(f"{args.output} is not a folder")
        if args.output == args.input:
            raise ValueError("Input and output should be different folders")

        for path_in in args.input.rglob("*.wav"):
            path_out = args.output / path_in.name
            separate_one_file(separator, path_in, path_out, args.src, args.mic)
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

        separate_one_file(separator, args.input, path_out, args.src, args.mic)
        print(f"{args.input} -> {path_out}")
