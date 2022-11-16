from pathlib import Path
import yaml
import torchaudio
import torch


def make_batch_array(lst):

    m = max([x.shape[-1] for x in lst])
    return torch.cat([x[None, :, :m] for x in lst], dim=0)


def read_samples(ref_mic=0):
    # choose and read the audio files
    samples_dir = Path(__file__).parent
    with open(samples_dir / "samples_list.yaml", "r") as f:
        samples = yaml.safe_load(f)

    mix_lst = []
    ref_lst = []
    for sample in samples:

        # the mixtures
        mix, fs_1 = torchaudio.load(samples_dir / sample["mix"])
        mix_lst.append(mix)

        # now load the references
        audio_ref_list = []
        for fn in sample["ref"]:
            audio, fs_2 = torchaudio.load(samples_dir / fn)
            assert fs_1 == fs_2
            audio_ref_list.append(audio[[ref_mic], :])

        ref = torch.cat(audio_ref_list, dim=0)
        ref_lst.append(ref)

    fs = fs_1

    mix = make_batch_array(mix_lst)
    ref = make_batch_array(ref_lst)

    return mix, ref
