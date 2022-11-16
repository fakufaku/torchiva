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
        sig = (sig * 2 ** 15).astype(np.int16)
        wavfile.write(str(base_dir / fn), fs, sig.T)

samples_list_fn = Path(__file__).parent / "samples_list.yaml"

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

        room = pra.ShoeBox(room_dim, fs=fs, max_order=10, materials=pra.Material(0.7))

        for src, sig in zip(source_loc, audios):
            room.add_source(src, signal=sig)

        for mic in mic_loc:
            room.add_microphone(mic)

        premix = room.simulate(return_premix=True)
        mix = room.mic_array.signals
        M = max(abs(premix).max(), abs(mix).max())
        premix *= 0.95 / M
        mix *= 0.95 / M

        abs(mix - premix.sum(axis=0)).max()

        save_files(base_dir, sample["ref"], fs, premix)
        save_files(base_dir, [sample["mix"]], fs, [mix])
