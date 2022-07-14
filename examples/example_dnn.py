# Copyright (c) 2022 Robin Scheibler, Kohei Saijo
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

import fast_bss_eval
import torch
import torchaudio
import torchiva
import yaml

import source_models
from dataloader import WSJ1SpatialDataset

REF_MIC = 0


def make_batch_array(lst, adjust="min"):

    if adjust == "max":
        m = max([x.shape[-1] for x in lst])
        batch = lst[0].new_zeros((len(lst), lst[0].shape[0], m))
        for i, example in enumerate(lst):
            batch[i, :, : example.shape[1]] = example
        return batch
    elif adjust == "min":
        m = min([x.shape[-1] for x in lst])
        return torch.cat([x[None, :, :m] for x in lst], dim=0)
    else:
        raise NotImplementedError()


if __name__ == "__main__":

    torch.manual_seed(0)

    parser = argparse.ArgumentParser(description="Separation example")

    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Location of dataset",
    )
    parser.add_argument(
        "model_path",
        type=Path,
        help="Model parameter path",
    )
    parser.add_argument(
        "-r",
        "--room",
        default=222,
        type=int,
        metavar="ROOMS",
        help="Room number (Sample number)",
    )
    parser.add_argument(
        "-n", "--n_iter", default=20, type=int, help="Number of iterations"
    )
    parser.add_argument("-m", "--mic", default=2, type=int, help="Number of mics")
    parser.add_argument("-s", "--src", default=2, type=int, help="Number of sources")

    args = parser.parse_args()

    print(f"Using rooms {args.room}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # get the data
    dataset = WSJ1SpatialDataset(
        args.dataset_dir / "dev93",
        shuffle_channels=False,
        noiseless=False,
        ref_is_reverb=True,
    )

    assert args.src <= args.mic

    # load the pre-trained model
    separator = torchiva.load_separator_model(
        args.model_path / "model_weights.ckpt",
        args.model_path / "model_config.yaml",
        n_iter=args.n_iter,
        n_src=args.src,
    )

    separator.to(device)
    mix, ref = dataset[args.room]
    mix, ref = mix.to(device), ref.to(device)

    with torch.no_grad():
        y = separator(mix[..., : args.src, :])

        # if args.n_src > n_ref, select most energetic n_ref sources
        if y.shape[-2] > ref.shape[-2]:
            y = torchiva.select_most_energetic(
                y, num=ref.shape[-2], dim=-2, dim_reduc=-1
            )

        m = min(ref.shape[-1], y.shape[-1])
        sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])

    print("\n==== Separation Results ====")
    print(f"n_iter: {args.n_iter:.0f},  n_chan: {args.mic:.0f},  n_src: {args.src:.0f}")
    print("SDR: ", sdr.cpu().numpy(), " | SIR: ", sir.cpu().numpy())
