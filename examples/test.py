# Copyright 2021 Robin Scheibler, Kohei Saijo
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of
# this software and associated documentation files (the "Software"), to deal in
# the Software without restriction, including without limitation the rights to
# use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies
# of the Software, and to permit persons to whom the Software is furnished to do
# so, subject to the following conditions:
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
import os
import re
import time
from pathlib import Path

import fast_bss_eval
import numpy as np
import torch as pt
# We will first validate the numpy backend
import torchiva
import yaml
from scipy.io import wavfile

from dataloader import WSJ1SpatialDataset, collator
from separation_model import WSJModel

DATA_DIR = Path("bss_speech_dataset/data")
DATA_META = DATA_DIR / "metadata.json"
REF_MIC = 0
RTOL = 1e-5


def get_paths(output, dataset_name, tag_dataset, dir_name):
    """
    Create all the relative paths
    """

    # output_dir = output / dataset_name / f"{tag_dataset}/{tags['algo']}_{tags['model']}"

    output_dir = output / dataset_name / f"{tag_dataset}/{dir_name}"

    # create the subdirectory for wav files
    output_wav_dir = output_dir / "wav"

    transcripts_file = output_dir / "asr_transcripts.csv"
    speakers_file = output_dir / "asr_speakers.csv"
    utterances_file = output_dir / "asr_utterances_files.csv"
    results_file = output_dir / "data.json"
    result_txt_file = output_dir / "result.txt"
    summary_file = output_dir / "summary.json"

    return (
        output_dir,
        output_wav_dir,
        transcripts_file,
        speakers_file,
        utterances_file,
        results_file,
        result_txt_file,
        summary_file,
    )


def make_batch_array(lst):

    m = np.min([x.shape[-1] for x in lst])
    return np.array([x[:, :m] for x in lst])


def data_to(data, device):
    if isinstance(data, np.ndarray):
        data = pt.from_numpy(data)
    return data.to(device)


def write_wav(filename, fs, data):
    if data.dtype != np.int16:
        m = np.abs(data).max()
        # if m > 1.0:
        # scale if necessary
        data = 0.95 * data / m
        data = (data * 2**15).astype(np.int16)
    wavfile.write(wav_filename, fs, data)


def table_2_file(filename, table, sort_keys=False):
    if sort_keys:
        table = sorted(table, key=lambda e: e[0])

    with open(filename, "w") as f:
        for fields in table:
            print(" ".join(fields), file=f)


def load_model(hparams_file, epoch=-1, eval=True):

    with open(hparams_file, "r") as f:
        hparams = yaml.load(f, Loader=yaml.FullLoader)

    # find the checkpoint
    ckpt_dir = hparams_file.parent / "checkpoints"

    for ckpt in ckpt_dir.iterdir():
        if "epoch=" + str(epoch) + "-" in str(ckpt):
            checkpoint = ckpt
            break

    # path -> str type
    checkpoint = str(checkpoint)
    hparams_file = str(hparams_file)

    # init model
    model = WSJModel.load_from_checkpoint(
        checkpoint_path=checkpoint,
        hparams_file=hparams_file,
        map_location=None,
        strict=False,
    )
    model = model.separator

    if eval:
        model.eval()

    return hparams, model


def load_ensemble_model(hparams_file, epochs, eval=True):

    for i, epoch in enumerate(epochs):
        hparams, model = load_model(hparams_file, epoch)
        state_dict = model.source_model.state_dict()
        if i == 0:
            ensemble_model_state_dict = state_dict
        else:
            for key in state_dict.keys():
                ensemble_model_state_dict[key] += state_dict[key]

    for key in ensemble_model_state_dict.keys():
        dtype = ensemble_model_state_dict[key].dtype
        ensemble_model_state_dict[key] = (
            ensemble_model_state_dict[key] / len(epochs)
        ).to(dtype)

    model.source_model.load_state_dict(ensemble_model_state_dict)

    if eval:
        model.eval()

    return hparams, model


if __name__ == "__main__":

    np.random.seed(0)

    parser = argparse.ArgumentParser(description="Run one algorithm over the dataset")
    parser.add_argument("--n_fft", default=1024, type=int, help="STFT FFT size")
    parser.add_argument("--hop", type=int, help="STFT hop length size")
    parser.add_argument(
        "--window",
        type=torchiva.Window,
        choices=torchiva.window_types,
        help="The STFT window type",
    )
    parser.add_argument(
        "-n", "--n_iter", default=15, type=int, help="Number of iterations of IVA"
    )
    parser.add_argument(
        "--n_chan", default=2, type=int, help="Number of channles used in separation"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        default="wsj1_2345_db/wsj1_2_mix_m2",
        help="Location of dataset",
    )

    # choose the algorithms
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--laplace", action="store_true", help="Location of mask hyperparameters"
    )
    group.add_argument(
        "--gauss", action="store_true", help="Location of mask hyperparameters"
    )
    group.add_argument(
        "--ilrma", action="store_true", help="Location of mask hyperparameters"
    )
    group.add_argument(
        "--iss-hparams", type=Path, help="Location of source model hyperparameters"
    )
    group.add_argument(
        "--ip2-hparams", type=Path, help="Location of source model hyperparameters"
    )
    group.add_argument(
        "--frontend-hparams", type=Path, help="Location of source model hyperparameters"
    )

    # choose the dataset split
    ds_split_group = parser.add_mutually_exclusive_group()
    ds_split_group.add_argument("--test", action="store_true", help="Use test set")
    ds_split_group.add_argument("--val", action="store_true", help="Use validation set")
    ds_split_group.add_argument("--train", action="store_true", help="Use training set")

    parser.add_argument(
        "--epoch",
        nargs="+",
        help="Epoch number to be evaluated. If multiple values are given, ensembled model is evaluated.",
    )

    parser.add_argument(
        "--noiseless", action="store_true", help="Use noiseless dataset"
    )
    parser.add_argument(
        "--ref_is_reverb", action="store_true", help="Use reverberant reference"
    )
    parser.add_argument(
        "--limit", type=int, help="Maximum number of samples to process"
    )
    parser.add_argument(
        "--output", default="results", type=Path, help="The output folder"
    )
    parser.add_argument(
        "--save-audios",
        action="store_true",
        help="Whether separated signals are saved as audio or not",
    )
    parser.add_argument(
        "--algo",
        default="tiss",
        type=str,
        help="Separation Algorithm e.g. ilrma_t_iss",
    )
    args = parser.parse_args()

    # STFT parameters
    if args.hop is None:
        args.hop = args.n_fft // 4

    print("n_fft:", args.n_fft, " | hop_length:", args.hop)

    tags = {}

    try:
        n_src = int(str(args.dataset)[-1])
    except ValueError:
        n_src = 2
    reset = False

    print("\n!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    print(f"Use {args.n_chan}ch / Extract {n_src} sources")
    print("!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!")

    if args.n_chan == 2:
        chan = [0, 3]
    elif args.n_chan == 3:
        chan = [0, 2, 4]
    elif args.n_chan == 4:
        chan = [0, 1, 3, 4]
    elif args.n_chan == 5:
        chan = [0, 1, 2, 3, 4]
    elif args.n_chan == 6:
        chan = [0, 1, 2, 3, 4, 5]

    # get the correct algorithm depending on input arguments
    if args.ilrma:
        separator = torchiva.nn.BSSSeparator(
            n_fft=args.n_fft,
            n_src=n_src,
            hop_length=args.hop,
            window=args.window,
            source_model=torchiva.NMFModel(),
            n_iter=args.n_iter,
            n_delay=0,
            n_taps=0,
            algo=args.algo,
        )
        reset = True

        tags["algo"] = "ilrma"
        tags["model"] = "nmf"
        algo_model = args.algo

    elif args.gauss:
        separator = torchiva.nn.BSSSeparator(
            n_fft=args.n_fft,
            n_src=n_src,
            hop_length=args.hop,
            window=args.window,
            source_model=torchiva.GaussModel(),
            n_iter=args.n_iter,
            n_delay=0,
            n_taps=0,
            wpe_n_iter=3,
            wpe_n_delay=3,
            wpe_n_taps=10,
            wpe_n_fft=512,
            dnn_nchan=args.n_chan,
            bss_nchan=args.n_chan,
            algo=args.algo,
        )

        tags["algo"] = "iva"
        tags["model"] = "gauss"
        algo_model = args.algo

    elif args.iss_hparams is not None:
        hparams, separator = load_ensemble_model(args.iss_hparams, args.epoch)

        separator.n_iter = args.n_iter
        separator.n_src = n_src
        separator.bss_nchan = args.n_chan
        separator.bss_n_iter = 0
        separator.dnn_nchan = args.n_chan

        separator.eval()

        tags["algo"] = hparams["config"]["algorithm"]
        tags["model"] = hparams["config"]["model"]["source_model"]["name"]
        algo_model = tags["algo"] + "-" + tags["model"]

    else:
        # default to the laplace model
        # hparams, separator = load_model(args.frontend_hparams, args.epoch)
        # separator.n_iter = args.n_iter

        separator = torchiva.nn.BSSSeparator(
            n_fft=args.n_fft,
            n_src=n_src,
            hop_length=args.hop,
            window=args.window,
            source_model=torchiva.LaplaceModel(),
            n_iter=args.n_iter,
            n_delay=0,
            n_taps=0,
            algo=args.algo,
        )

        tags["algo"] = "iva"
        tags["model"] = "laplace"
        algo_model = tags["algo"] + "_" + tags["model"]

    if args.test:
        tag_dataset = "eval92"
    elif args.train:
        tag_dataset = "si284"
    else:
        # default is using the validation dataset
        tag_dataset = "dev93"

    # get the data
    dataset = WSJ1SpatialDataset(
        args.dataset / tag_dataset,
        shuffle_channels=False,
        noiseless=args.noiseless,
        ref_is_reverb=args.ref_is_reverb,
    )

    # convert to pytorch tensor if necessary

    print(f"Is GPU available ? {pt.cuda.is_available()}")
    device = pt.device("cuda:0" if pt.cuda.is_available() else "cpu")
    print(f"Using {device}")
    separator = separator.to(device)

    # prepare the output folders
    os.makedirs(args.output, exist_ok=True)

    (
        output_dir,
        output_wav_dir,
        transcripts_file,
        speakers_file,
        utterances_file,
        results_file,
        results_txt_file,
        summary_json_file,
    ) = get_paths(args.output, args.dataset.name, tag_dataset, algo_model)

    # create a directory as needed
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs(output_wav_dir, exist_ok=True)

    print(output_dir)

    f = open(results_txt_file, "w")

    # empty lists for ASR related stuff
    transcripts_list = []
    speakers_list = []
    utterances_list = []

    # save in a list of results
    results = []

    print("Len : ", len(dataset))
    sdr_total = 0
    sir_total = 0
    mixsdr_total = 0
    mixsir_total = 0

    for idx, (mix, ref) in enumerate(dataset):

        mix = data_to(mix, device)
        ref = data_to(ref, device)
        mixinfo = dataset.get_mixinfo(idx)
        data_id = mixinfo["data_id"]
        mix_id = f"{tag_dataset}_{data_id}"
        fs = mixinfo["wav_frame_rate_mixed"]

        mix = mix[chan]

        with pt.no_grad():
            t1 = time.perf_counter()

            ch = ref.shape[-2]

            # Separation
            y = separator(mix, reset=reset)

            t2 = time.perf_counter()

            m = np.minimum(ref.shape[-1], y.shape[-1])

            # scale invaliant metric
            sdr, sir, sar, perm = fast_bss_eval.bss_eval_sources(ref[:, :m], y[:, :m])

            mix_to_calc = pt.tile(mix[REF_MIC, None], (ch, 1))
            mix_sdr, mix_sir, mix_sar, _ = fast_bss_eval.bss_eval_sources(
                ref[:, :m], mix_to_calc[:, :m]
            )

            # get the number of iterations for iterative methods
            if isinstance(separator, torchiva.nn.BSSSeparator):
                tag_n_iter = separator.n_iter

            else:
                tag_n_iter = 0

            t3 = time.perf_counter()

        mix = mix.cpu().numpy()
        ref = ref.cpu().numpy()
        y = y.detach().cpu().numpy()

        perm = perm.detach().cpu().numpy()
        mix_to_calc = mix_to_calc.detach().cpu().numpy()

        # save the separated channels to feed to ASR later

        gen = zip(
            perm,
            mixinfo["speaker_id"],
            mixinfo["utterance_id"],
            mixinfo["transcript_espnet"],
        )
        for src_id, (out_ch, spk_id, utt_id, transcript) in enumerate(gen):
            utterance_id = f"{mix_id}_{src_id}"
            wav_filename = output_wav_dir / f"{tag_dataset}_{data_id}_ch{src_id}.wav"

            results.append(
                {
                    "utterance_id": utterance_id,
                    "channels": mix.shape[-2],
                    "algo": tags["algo"],
                    "model": tags["model"],
                    "algo_model": algo_model,
                    "n_iter": tag_n_iter,
                    "sdr": float(sdr[src_id] - mix_sdr[src_id]),
                    "sir": float(sir[src_id] - mix_sir[src_id]),
                    "sep_time": t2 - t1,
                    "eval_time": t3 - t2,
                    "split": tag_dataset,
                }
            )

            speakers_list.append([utterance_id, spk_id])
            utterances_list.append([utterance_id, str(wav_filename.resolve())])
            transcripts_list.append([utterance_id, transcript])

            t4 = time.perf_counter()
            write_wav(wav_filename, fs, y[out_ch])

        basic_info = f"{tags['algo']} {tags['model']} {idx:3d} {data_id}  sep_tim {t2 - t1:.3f} s  eval_time {t4 - t2:.3f} s  "
        sdr_info = f"SDR {pt.mean(sdr):5.2f}  "
        sir_info = f"SIR {pt.mean(sir):5.2f}  "

        info = basic_info + sdr_info + sir_info
        print(info)
        info = info + sir_info + "\n"
        f.write(info)

        sdr_total += pt.mean(sdr)
        sir_total += pt.mean(sir)
        mixsdr_total += pt.mean(mix_sdr)
        mixsir_total += pt.mean(mix_sir)

        if args.limit is not None and idx == args.limit:
            break

    total_basic_info = f"\n{tags['algo']} {tags['model']} n_iter {args.n_iter:.0f}  n_fft {args.n_fft:.0f}  "
    total_sdr_sir_info = f"SDR {sdr_total/(idx+1):5.2f}  SIR {sir_total/(idx+1):5.2f}  "
    total_result = total_basic_info + total_sdr_sir_info
    print(total_result)

    # f = open(results_txt_file, 'w')
    f.write(total_result + "\n")
    f.close()

    sdr_total = float(sdr_total / (idx + 1))
    sir_total = float(sir_total / (idx + 1))
    mixsdr_total = float(mixsdr_total / (idx + 1))
    mixsir_total = float(mixsir_total / (idx + 1))

    with open(summary_json_file, "w") as f:
        json.dump(
            {
                "sdr": sdr_total,
                "sir": sir_total,
                "mix_sdr": mixsdr_total,
                "mix_sir": mixsir_total,
            },
            f,
            indent=2,
        )

    # HACK!! Currently, ESPNET evaluation script does not handle well having different speakers.
    # => replace all the speaker ids with the first speaker id
    spkr_id_1 = speakers_list[0][1]
    for entry in speakers_list:
        entry[1] = spkr_id_1

    # Now save everything
    table_2_file(transcripts_file, transcripts_list, sort_keys=True)
    table_2_file(utterances_file, utterances_list, sort_keys=True)
    table_2_file(speakers_file, speakers_list, sort_keys=True)
    with open(results_file, "w") as f:
        json.dump(results, f)
