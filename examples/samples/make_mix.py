import yaml
from pathlib import Path
import pyroomacoustics as pra
import numpy as np
from scipy.io import wavfile


def read_files(base_dir, filelist):
    audios = []
    for fn in filelist:
        fs, audio = wavfile.read(base_dir / fn)
        audios.append(audio)
    return fs, audios


def save_files(base_dir, filelist, fs, signals):
    for fn, sig in zip(filelist, signals):
        sig = (sig * 2**15).astype(np.int16)
        wavfile.write(str(base_dir / fn), fs, sig.T)


samples_list_fn = Path(__file__).parent / "samples_list.yaml"


def simulate(
    room_dim, source_loc, mic_loc, fs, audios, max_order, energy_absorption=0.9
):
    room = pra.ShoeBox(
        room_dim, fs=fs, max_order=max_order, materials=pra.Material(energy_absorption)
    )

    for src, sig in zip(source_loc, audios):
        room.add_source(src, signal=sig)

    for mic in mic_loc:
        room.add_microphone(mic)

    premix = room.simulate(return_premix=True)

    return premix


def pad_to_max(*arrays):

    max_len = max([a.shape[-1] for a in arrays])

    out = []
    for a in arrays:
        new = np.zeros(a.shape[:-1] + (max_len,))
        new[..., : a.shape[-1]] = a
        out.append(new)

    return tuple(out)


def scale_(scale, *arrays):
    for a in arrays:
        a *= scale


def max_abs(*arrays):
    return max([abs(a).max() for a in arrays])


if __name__ == "__main__":
    np.random.seed(0)

    with open(samples_list_fn, "r") as f:
        samples_list = yaml.safe_load(f)

    base_dir = samples_list_fn.parent

    for sample in samples_list:
        n_chan = sample["n_chan"]

        fs, audios = read_files(base_dir, sample["dry"])

        room_dim = np.random.rand(3) * 3 + 3
        source_loc = np.random.rand(2, 3) * room_dim
        mic_loc = room_dim / 2.0 + 0.1 * (np.random.rand(n_chan, 3) - 0.5)

        max_order = 20
        energy_absorption = 0.35

        premix_dry = simulate(
            room_dim,
            source_loc,
            mic_loc,
            fs,
            audios,
            max_order=0,
            energy_absorption=energy_absorption,
        )
        premix_wet = simulate(
            room_dim,
            source_loc,
            mic_loc,
            fs,
            audios,
            max_order=max_order,
            energy_absorption=energy_absorption,
        )
        mix_wet = premix_wet.sum(axis=0)

        premix_dry, premix_wet, mix_wet = pad_to_max(premix_dry, premix_wet, mix_wet)

        M = max_abs(mix_wet, premix_dry, premix_wet)
        scale_(0.95 / M, mix_wet, premix_dry, premix_wet)

        save_files(base_dir, sample["ref_anecho"], fs, premix_dry)
        save_files(base_dir, sample["ref_reverb"], fs, premix_wet)
        save_files(base_dir, [sample["mix_reverb"]], fs, [mix_wet])
