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
from pathlib import Path
import torch
import torchaudio
import warnings

import torchiva
import fast_bss_eval

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
        return torch.cat([x[None, :, :m] for x in lst], dim=0)
    else:
        raise NotImplementedError()


if __name__ == "__main__":

    torch.manual_seed(0)

    source_models = list(torchiva.models.source_models.keys())

    parser = argparse.ArgumentParser(description="Separation example")

    parser.add_argument(
        "dataset_dir",
        type=Path,
        help="Location of dataset",
    )
    parser.add_argument(
        "algorithm",
        type=str,
        choices=["overiss_t", "overiva_ip", "ip2", "five"],
        help="BSS algorithm",
    )
    parser.add_argument(
        "-r",
        "--room",
        default="00221",
        metavar="ROOMS",
        type=str,
        help="Room number (Sample number)",
    )
    parser.add_argument("--n_fft", type=int, default=4096, help="STFT FFT size")
    parser.add_argument("--hop", type=int, default=None, help="STFT hop length size")
    parser.add_argument(
        "--n_iter", default=20, type=int, help="Number of iterations"
    )
    parser.add_argument("--delay", type=int, default=0, help="Delay in dereverberation")
    parser.add_argument("--tap", type=int, default=0, help="Tap length in dereverberation")
    parser.add_argument("--n_src", type=int, default=2, help="Number of sources in a mixture")
    parser.add_argument("--n_chan", type=int, default=None, help="Number of channels used for separation")

    parser.add_argument(
        "-d",
        "--source_model",
        default=source_models[1],
        #choices=source_models,
        choices=["laplace", "gauss", "nmf"],
        type=str,
        help="Source model, default: gauss",
    )
    
    args = parser.parse_args()

    metafilename = args.dataset_dir / "dev93" / "mixinfo_noise.json"
    with open(metafilename, "r") as f:
        metadata = json.load(f)

    info = metadata[args.room]
    rt60 = info["rir_info_t60"]
    print(f"Using rooms {args.room} with RT60 {rt60:.4f}")

    if args.n_chan is None:
        args.n_chan = args.n_src

    if args.algorithm != "overiss_t" and args.tap > 0:
        warnings.warn(f"``tap={args.tap}`` is specified, but dereverberation is not performed in ``{args.algorithm}``.")



    # choose and read the audio files
    # can be changed to any multi-channel mixture and corresponding reference signals

    # mix: shape (n_chan, n_sample)
    mix, fs = torchaudio.load(args.dataset_dir / (Path("").joinpath(*Path(info['wav_mixed_noise_reverb']).parts[-4:])))
    
    if args.delay==0 and args.tap==0:
        # reverberant clean references
        ref1, fs = torchaudio.load(args.dataset_dir / (Path("").joinpath(*Path(info['wav_dpath_image_reverberant'][0]).parts[-4:])))
        ref2, fs = torchaudio.load(args.dataset_dir / (Path("").joinpath(*Path(info['wav_dpath_image_reverberant'][1]).parts[-4:])))
    else:
        # anechoic clean references
        ref1, fs = torchaudio.load(args.dataset_dir / (Path("").joinpath(*Path(info['wav_dpath_image_anechoic'][0]).parts[-4:])))
        ref2, fs = torchaudio.load(args.dataset_dir / (Path("").joinpath(*Path(info['wav_dpath_image_anechoic'][1]).parts[-4:])))
    
    # ref: shape (n_src, n_sample)
    ref = torch.stack((ref1[REF_MIC], ref2[REF_MIC]),dim=0)

    stft = torchiva.STFT(
        n_fft=args.n_fft,
        hop_length=args.hop, 
    )

    source_model = torchiva.models.source_models[args.source_model]

    if args.algorithm == "overiss_t":
        separator = torchiva.OverISS_T(
            n_iter=args.n_iter,
            n_taps=args.tap,
            n_delay=args.delay,
            n_src=args.n_src,
            model=source_model,
            proj_back_mic=REF_MIC,
            eps=1e-5,
        )
    elif args.algorithm == "overiva_ip":
        separator = torchiva.OverIVA_IP(
            n_iter=args.n_iter,
            n_src=args.n_src,
            model=source_model,
            proj_back_mic=REF_MIC,
            eps=1e-5,
        )
    elif args.algorithm == "ip2":
        assert (args.n_chan==2 and args.n_src==2)
        separator = torchiva.AuxIVA_IP2(
            n_iter=args.n_iter,
            model=source_model,
            proj_back_mic=REF_MIC,
            eps=1e-5,
        )
    elif args.algorithm == "five":
        if args.n_src != 1:
            warnings.warn(f"``n_src={args.tap}`` is specified, but ``five`` extracts only 1 source.")

        separator = torchiva.FIVE(
            n_iter=args.n_iter,
            model=source_model,
            proj_back_mic=REF_MIC,
            eps=1e-5,
            n_power_iter=None,
        )


    X = stft(mix[..., :args.n_chan, :])
    Y = separator(X)
    y = stft.inv(Y)

    # if args.n_src > n_ref, select most energetic n_ref sources
    if y.shape[-2] > ref.shape[-2]:
        y = torchiva.select_most_energetic(y, num=ref.shape[-2], dim=-2, dim_reduc=-1)

    m = min(ref.shape[-1], y.shape[-1])
    sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])

    print("\n==== Separation Results ====")
    print(f"Algo: {args.algorithm.upper()},  Model: {args.source_model},  n_iter: {args.n_iter:.0f},  n_chan: {args.n_chan:.0f},  n_src: {args.n_src:.0f}")
    print("SDR: ", sdr.to('cpu').numpy(), " | SIR: ", sir.to('cpu').numpy())