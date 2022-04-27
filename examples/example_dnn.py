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

import source_models
import torchiva
import fast_bss_eval

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


def load_source_model(model_config, model_param_path):
    model = source_models.get_model(**model_config)
    model.load_state_dict(torch.load(model_param_path))
    model.eval()

    return model

if __name__ == "__main__":

    torch.manual_seed(0)

    parser = argparse.ArgumentParser(description="Separation example")

    parser.add_argument(
        "config",
        type=Path,
        help="Configuration file path",
    )
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
        default="00222",
        metavar="ROOMS",
        type=str,
        help="Room number (Sample number)",
    )

    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    metafilename = args.dataset_dir / "dev93" / "mixinfo_noise.json"
    with open(metafilename, "r") as f:
        metadata = json.load(f)

    info = metadata[args.room]
    rt60 = info["rir_info_t60"]
    print(f"Using rooms {args.room} with RT60 {rt60:.4f}")

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # choose and read the audio files
    # can be changed to any multi-channel mixture and corresponding reference signals
    # mix: shape (n_chan, n_sample)
    mix, fs = torchaudio.load(args.dataset_dir / (Path("").joinpath(*Path(info['wav_mixed_noise_reverb']).parts[-4:])))
    
    if config["training"]["ref_is_reverb"]:
        # reverberant clean references
        ref1, fs = torchaudio.load(args.dataset_dir / (Path("").joinpath(*Path(info['wav_dpath_image_reverberant'][0]).parts[-4:])))
        ref2, fs = torchaudio.load(args.dataset_dir / (Path("").joinpath(*Path(info['wav_dpath_image_reverberant'][1]).parts[-4:])))
    else:
        # anechoic clean references
        ref1, fs = torchaudio.load(args.dataset_dir / (Path("").joinpath(*Path(info['wav_dpath_image_anechoic'][0]).parts[-4:])))
        ref2, fs = torchaudio.load(args.dataset_dir / (Path("").joinpath(*Path(info['wav_dpath_image_anechoic'][1]).parts[-4:])))
    
    # ref: shape (n_src, n_sample)
    ref = torch.stack((ref1[REF_MIC], ref2[REF_MIC]),dim=0)

    source_model = load_source_model(config["model"]["source_model"], args.model_path)

    n_iter = config["model"]["n_iter"]
    n_chan, n_src = config["training"]["n_chan"], config["training"]["n_src"]
    assert n_src <= n_chan

    separator = torchiva.nn.BSSSeparator(
        config["model"]["n_fft"],
        n_iter,
        hop_length=config["model"]["hop_length"],
        n_taps=config["model"]["n_taps"],
        n_delay=config["model"]["n_delay"],
        n_src=n_src,
        algo=config["algorithm"],
        source_model=source_model, 
        proj_back_mic=config["model"]["ref_mic"],
        use_dmc=config["model"]["use_dmc"],
        n_power_iter=config["model"]["n_power_iter"],
    )

    separator.to(device)
    mix, ref = mix.to(device), ref.to(device)


    with torch.no_grad():
        y = separator(mix[..., :n_chan, :])

        # if args.n_src > n_ref, select most energetic n_ref sources
        if y.shape[-2] > ref.shape[-2]:
            y = torchiva.select_most_energetic(y, num=ref.shape[-2], dim=-2, dim_reduc=-1)

        m = min(ref.shape[-1], y.shape[-1])
        sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])

    print("\n==== Separation Results ====")
    print(f"n_iter: {n_iter:.0f},  n_chan: {n_chan:.0f},  n_src: {n_src:.0f}")
    print("SDR: ", sdr.cpu().numpy(), " | SIR: ", sir.cpu().numpy())