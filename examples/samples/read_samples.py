from pathlib import Path
import yaml
import torchaudio
import torch


def make_batch_array(lst):

    m = min([x.shape[-1] for x in lst])
    return torch.cat([x[None, :, :m] for x in lst], dim=0)


def read_audio(fn_list, samples_dir, ref_mic=None, fs_expected=None):
    audio_ref_list = []
    for fn in fn_list:
        audio, fs_2 = torchaudio.load(samples_dir / fn)
        if fs_expected is not None:
            assert fs_expected == fs_2
        if ref_mic is None:
            audio_ref_list.append(audio)
        else:
            audio_ref_list.append(audio[[ref_mic], :])

    if ref_mic is None:
        out = torch.stack(audio_ref_list, dim=0)
    else:
        out = torch.cat(audio_ref_list, dim=0)

    return out


def read_samples(ref_mic=None):
    # choose and read the audio files
    samples_dir = Path(__file__).parent
    with open(samples_dir / "samples_list.yaml", "r") as f:
        samples = yaml.safe_load(f)

    mix_lst = []
    ref_wet_lst = []
    ref_dry_lst = []
    for sample in samples:

        # the mixture
        mix, fs = torchaudio.load(samples_dir / sample["mix_reverb"])
        mix_lst.append(mix)

        # now load the references
        ref_wet_lst.append(read_audio(sample["ref_reverb"], samples_dir, ref_mic, fs))
        ref_dry_lst.append(read_audio(sample["ref_anecho"], samples_dir, ref_mic, fs))

    mix = make_batch_array(mix_lst)
    ref_wet = make_batch_array(ref_wet_lst)
    ref_dry = make_batch_array(ref_dry_lst)

    return mix, ref_wet, ref_dry
