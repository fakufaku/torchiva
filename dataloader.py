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


import json
import math
from pathlib import Path
from typing import List, Optional, Union

import numpy as np
import torch
import torchaudio
from scipy.io import wavfile
import random


def compute_interval(n_target, n_mix, n_originals, n_offsets):
    """
    Compute an interval containing all the sources

    Parameters
    ----------
    n_target:
        target number samples
    n_mix:
        number of samples in mixture
    n_originals
        number of samples of the sources
    n_offsets
        offset of the sources in the mixtures
    """

    if n_target >= n_mix:
        return 0, n_mix

    n_mix = np.array(n_mix)
    n_originals = np.array(n_originals)
    n_offsets = np.array(n_offsets)

    # left is the sample at which starts the source that starts last
    # right is thee sample at which terminates the first source to terminate
    left, right = np.max(n_offsets), np.min(n_offsets + n_originals)

    # the midpoint between left and right should be the center of the
    # target interval
    midpoint = 0.5 * (left + right)

    # the start and end of interval
    start = midpoint - n_target // 2
    end = start + n_target

    # handle border effects
    if start < 0:
        return 0, n_target
    elif end >= n_mix:
        return n_mix - n_target, n_mix
    else:
        return int(start), int(end)


class BSSSpeechDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        root_dir: Union[Path, str],
        room_list: List,
        snr: Optional[float] = 40.0,
        ref_mic: Optional[int] = 0,
    ):
        super().__init__()

        self.root_dir = Path(root_dir)
        self.room_list = room_list
        self.ref_mic = ref_mic
        self.snr = snr
        self._data_index = 0

    def __len__(self):
        return len(self.room_list)

    def __getitem__(self, idx):

        room = self.room_list[idx]

        # torchaudio loads the data, converts to float, and normalize to [-1, 1] range
        #audio_mix, fs_1 = torchaudio.load_wav(self.root_dir / room["mix_filename"])
        #audio_ref, fs_2 = torchaudio.load_wav(
        #    self.root_dir / room["src_filenames"][self.ref_mic]
        #)
        audio_mix, fs_1 = torchaudio.load(self.root_dir / room["mix_filename"])
        audio_ref, fs_2 = torchaudio.load(
            self.root_dir / room["src_filenames"][self.ref_mic]
        )

        audio_mix /= 2 ** 15
        audio_ref /= 2 ** 15

        assert fs_1 == fs_2

        # add some noise
        sigma_src = torch.std(audio_mix)
        sigma_n = sigma_src * 10 ** (-self.snr / 20)
        audio_mix += torch.zeros(audio_mix.shape).normal_() * sigma_n

        return audio_mix, audio_ref


class WSJ1SpatialDataset(torch.utils.data.Dataset):
    """
    Dataloader for the WSJ1-spatialized datasets for multichannel

    Parameters
    ----------
    metafilename: pathlib.Path or str
        the path to the mixinfo.json file containing the dataset metadata
    max_len_s: float, optional
        the length in seconds of the samples
    ref_mic: int optional
        the microphone to use for the scaling reference, if not provided
        it is chosen at random
    shuffle_channels: bool, optional
        if set to True (default), the channels of the microphones will be
        shuffled at random
    ref_is_reverb: bool, optional
        if set to True (default), the reverberant clean signal is used
        as a reference, if False, the anechoic clean signal is used
    noiseless: bool, optional
        if set to False (default), use the noisy mixture, if true, use the
        noiseless mixture
    max_n_samples: int, optional
    """

    def __init__(
        self,
        dataset_location: Union[Path, str],
        max_len_s: Optional[float] = None,
        ref_mic: Optional[int] = 0,
        shuffle_channels: Optional[bool] = True,
        shuffle_ref: Optional[bool] = False,
        ref_is_reverb: Optional[bool] = True,
        noiseless: Optional[bool] = False,
        max_n_samples: Optional[int] = None,
        return_mic_position: Optional[bool] = False,
    ):
        super().__init__()

        self.dataset_location = Path(dataset_location)
        self.metafilename = self.dataset_location / "mixinfo_noise.json"
        self.max_len_s = max_len_s
        self.ref_mic = ref_mic
        self.shuffle_channels = shuffle_channels
        self.shuffle_ref = shuffle_ref
        self.ref_is_reverb = ref_is_reverb
        self.noiseless = noiseless
        self.return_mic_position = return_mic_position

        # open the metadata and find the dataset path
        with open(self.metafilename, "r") as f:
            # the metadata is stored as a dict, but a list is preferable
            self.metadata = list(json.load(f).values())

        # we truncate the dataset if required
        if max_n_samples is not None:
            self.metadata = self.metadata[:max_n_samples]
        
    def __len__(self):
        return len(self.metadata)

    def get_mixinfo(self, idx):
        return self.metadata[idx]

    def __getitem__(self, idx):

        room = self.metadata[idx]

        if self.noiseless:
            mix_fn = Path(room["wav_dpath_mixed_reverberant"])
        else:
            mix_fn = Path(room["wav_mixed_noise_reverb"])

        # hack to get relative path from dataset directory
        mix_fn = Path("").joinpath(*mix_fn.parts[-3:])

        if self.ref_is_reverb:
            ref_fns_list = room["wav_dpath_image_reverberant"]
        else:
            ref_fns_list = room["wav_dpath_image_anechoic"]
        ref_fns = [Path(p) for p in ref_fns_list]

        # hack to get relative path from dataset directory
        ref_fns = [Path("").joinpath(*fn.parts[-3:]) for fn in ref_fns]

        # load the mixture audio
        # torchaudio loads the data, converts to float, and normalize to [-1, 1] range
        audio_mix, fs_1 = torchaudio.load(self.dataset_location / mix_fn)

        # now we know the number of channels
        n_channels = audio_mix.shape[0]

        # randomly shuffle the order of the channels in the mixture if required
        if self.shuffle_channels:
            p = torch.randperm(n_channels)
        else:
            p = torch.arange(n_channels)

        if not self.shuffle_ref:
            # the reference mic needs to be picked according the shuffled order
            ref_mic = p[self.ref_mic]
        else:
            # pick any of the channels as reference
            ref_mic = torch.randint(n_channels, size=(1,))[0]

        # now load the references
        audio_ref_list = []
        for fn in ref_fns:
            audio, fs_2 = torchaudio.load(self.dataset_location / fn)

            assert fs_1 == fs_2

            audio_ref_list.append(audio[ref_mic, None, :])
        

        audio_ref = torch.cat(audio_ref_list, dim=0)
        
        audio_mix = audio_mix[p]

        mic_center = room["rir_info_array_center"]
        mic_position = room["rir_info_microphone_position"]
        speaker_position = room["rir_info_speaker_position"]

        mic_position = torch.tensor(mic_position)
        mic_position = mic_position[p]

        if self.max_len_s is None:
            if self.return_mic_position:
                return audio_mix, audio_ref, mic_center, mic_position, speaker_position
            else:
                return audio_mix, audio_ref
        else:
            # the length of the different signals
            n_target = int(fs_1 * self.max_len_s)
            n_originals = room["wav_n_samples_original"]
            n_offsets = room["wav_offset"]
            n_mix = audio_mix.shape[-1]

            # compute an interval that has all sources in it
            s, e = compute_interval(n_target, n_mix, n_originals, n_offsets)

            mean_powers = torch.mean(audio_ref[...,s:e],dim=-1)
            for mean_power in mean_powers:
                if mean_power == 0:
                    #audio_mix, audio_ref = self.__getitem__(idx=random.randint(0,self.__len__()-1))
                    #return audio_mix, audio_ref
                    return self.__getitem__(idx=random.randint(0,self.__len__()-1))

            if self.return_mic_position:
                return audio_mix[..., s:e], audio_ref[..., s:e], mic_center, mic_position, speaker_position
            else:
                return audio_mix[..., s:e], audio_ref[..., s:e]

            #return (audio_mix[..., s:e], audio_ref[..., s:e])



class LibriCSSDataset(torch.utils.data.Dataset):
    """
    Dataloader for the LibriCSS datasets for multichannel

    Parameters
    ----------
    metafilename: pathlib.Path or str
        the path to the mixinfo.json file containing the dataset metadata
    max_len_s: float, optional
        the length in seconds of the samples
    shuffle_channels: bool, optional
        if set to True (default), the channels of the microphones will be
        shuffled at random
    """

    def __init__(
        self,
        dataset_location: Union[Path, str],
        max_len_s: Optional[float] = None,
        shuffle_channels: Optional[bool] = True,
        stage: Optional[int] = 'train',
        eval_with_wsj: Optional[bool] = False,
    ):
        super().__init__()

        self.dataset_location = Path(dataset_location)
        self.max_len_s = max_len_s
        self.shuffle_channels = shuffle_channels
        self.eval_with_wsj = eval_with_wsj

        if 'wsj1_6ch' in str(self.dataset_location):
            self.all_wavs, self.all_mic_positions = self.get_all_wavs_wsj6ch(self.dataset_location, stage)
        else:
            self.all_wavs = self.get_all_wavs(self.dataset_location, stage)
            if self.eval_with_wsj:
                self.wsj_mix, self.wsj_mic_positions, self.wsj_ref = self.get_all_wavs_wsj6ch(Path('wsj1_6ch'), stage, return_ref=True)

        if self.eval_with_wsj and stage=='valid':
            print(stage + ": " + str(len(self.wsj_mix)))
        else:
            print(stage + ": " + str(len(self.all_wavs)))

    def __len__(self):
        if self.eval_with_wsj:
            return len(self.wsj_mix)
        else:
            return len(self.all_wavs)

    def get_mixinfo(self, idx):
        return self.metadata[idx]

    def get_mic_position(self, num_mic=7):
        # !!! Caution !!!
        # height is set to 0

        # center mic is set to origin
        mic_position = [[0,0,0]]

        deg_to_rad = math.pi / 180
        radius = 4.25 / 100
        for i in range(num_mic-1):
            x = math.cos((30+60*i)*deg_to_rad)*radius
            y = math.sin((30+60*i)*deg_to_rad)*radius
            mic_position.append([x,y,0])
        
        #mic_position[1:] = mic_position[1:][::-1]
        
        return mic_position

    # if we use libricss dataset
    def get_all_wavs(self, data_path, stage, dev_session=0):
        dataset = []
        datadirs = list(data_path.iterdir())
        for datadir in datadirs:
            # remove utterance_transcription.txt
            if datadir.is_file():
                continue
            # separate training and development set depending on the stage
            if stage=='train' and 'session'+str(dev_session) in str(datadir.stem):
                continue
            elif stage=='valid' and not 'session'+str(dev_session) in str(datadir.stem):
                continue
            dataset += list(datadir.iterdir())

        return dataset

    # if we  use wsj1_6ch dataset
    def get_all_wavs_wsj6ch(self, data_path, stage, return_ref=False):
        dataset = []
        mic_position = []
        ref = []
        if stage == 'train':
            data_path = data_path / 'si284'
        elif stage == 'valid':
            data_path = data_path / 'dev93'
            
        with open(data_path/'mixinfo_noise.json', "r") as f:
            mixinfo = list(json.load(f).values())
        for m in mixinfo:
            mix_fn = Path(m["wav_mixed_noise_reverb"])
            mix_fn = Path("").joinpath(*mix_fn.parts[-3:])
            mix_fn = data_path / mix_fn
            dataset.append(mix_fn)
            mic_position.append(m["rir_info_microphone_position"])

            if return_ref:
                ref_fns_list = m["wav_dpath_image_reverberant"]
                ref_fns = [Path(p) for p in ref_fns_list]
                ref_fns = [data_path / (Path("").joinpath(*fn.parts[-3:])) for fn in ref_fns]
                ref.append(ref_fns)

        if return_ref:
            return dataset, mic_position, ref
        else:
            return dataset, mic_position

    def __getitem__(self, idx):

        if not self.eval_with_wsj:
            mix_fn = self.all_wavs[idx]

            audio_mix, fs = torchaudio.load(mix_fn)
            audio_mix = audio_mix / audio_mix.max() * 0.95

            # now we know the number of channels
            n_channels = audio_mix.shape[0]

            # randomly shuffle the order of the channels in the mixture if required
            if self.shuffle_channels:
                p = torch.randperm(n_channels)
            else:
                p = torch.arange(n_channels)
            
            audio_mix = audio_mix[p]

            if 'wsj1_6ch' in str(self.dataset_location):
                mic_position = torch.tensor(self.all_mic_positions[idx])
            else:
                mic_position = torch.tensor(self.get_mic_position())
                
            mic_position = mic_position[p]

            if self.max_len_s is None:
                return audio_mix, mic_position

            else:
                n_target = int(fs * self.max_len_s)
                if audio_mix.shape[-1] < n_target:
                    offset = n_target - audio_mix.shape[-1]
                    s = random.randint(0, offset)
                    zeros = torch.zeros((audio_mix.shape[0], offset))
                    audio_mix = torch.cat((zeros[...,:s], audio_mix, zeros[..., s:]), dim=-1)
                else:
                    s = random.randint(0,audio_mix.shape[-1]-n_target)
                    e = s + n_target
                    audio_mix = audio_mix[..., s:e]

                return audio_mix, mic_position

        # validate with wsj dataset
        else:
            mix_fn = self.wsj_mix[idx]
            ref_fn = self.wsj_ref[idx]

            audio_mix, fs = torchaudio.load(mix_fn)

            audio_ref1, _ = torchaudio.load(ref_fn[0])
            audio_ref2, _ = torchaudio.load(ref_fn[1])

            #audio_ref1 /= audio_mix.max() * 0.95
            #audio_ref2 /= audio_mix.max() * 0.95
            #audio_mix /= audio_mix.max() * 0.95

            # now we know the number of channels
            n_channels = audio_mix.shape[0]

            audio_ref1, audio_ref2 =audio_ref1[0], audio_ref2[0]
            audio_ref = torch.stack((audio_ref1, audio_ref2),dim=0)
            
            mic_position = torch.tensor(self.wsj_mic_positions[idx])

            return audio_mix, audio_ref, mic_position



class LibriCSSDataset_DOA(torch.utils.data.Dataset):
    """
    Dataloader for the LibriCSS datasets for multichannel

    Parameters
    ----------
    metafilename: pathlib.Path or str
        the path to the mixinfo.json file containing the dataset metadata
    max_len_s: float, optional
        the length in seconds of the samples
    shuffle_channels: bool, optional
        if set to True (default), the channels of the microphones will be
        shuffled at random
    """

    def __init__(
        self,
        dataset_location: Union[Path, str],
        max_len_s: Optional[float] = None,
        shuffle_channels: Optional[bool] = True,
        stage: Optional[int] = 'train',
        eval_with_wsj: Optional[bool] = False,
        ref_is_reverb: Optional[bool] = True,
    ):
        super().__init__()

        self.dataset_location = Path(dataset_location)
        self.max_len_s = max_len_s
        self.shuffle_channels = shuffle_channels
        self.eval_with_wsj = eval_with_wsj
        self.ref_is_reverb = ref_is_reverb

        if stage == 'train':
            with open(self.dataset_location/'libricss_dataset.json', "r") as f:
                self.all_wavs = list(json.load(f))
                
        else:
            self.wsj_mix, self.wsj_mic_positions, self.wsj_ref = self.get_all_wavs_wsj6ch(
                Path('wsj1_6ch_jutopia'), stage, return_ref=True
            )

        if stage=='valid':
            print(stage + ": " + str(len(self.wsj_mix)))
        else:
            print(stage + ": " + str(len(self.all_wavs)))

    def __len__(self):
        if self.eval_with_wsj:
            return len(self.wsj_mix)
        else:
            return len(self.all_wavs)

    def get_mixinfo(self, idx):
        return self.metadata[idx]

    def get_mic_position(self, num_mic=7):
        # center mic is set to origin
        mic_position = [[0,0,0]]

        deg_to_rad = math.pi / 180
        radius = 4.25 / 100
        for i in range(num_mic-1):
            x = math.cos((30+60*i)*deg_to_rad)*radius
            y = math.sin((30+60*i)*deg_to_rad)*radius
            mic_position.append([x,y,0])
        
        return mic_position

    # if we  use wsj1_6ch dataset
    def get_all_wavs_wsj6ch(self, data_path, stage, return_ref=False):
        dataset = []
        mic_position = []
        ref = []
        if stage == 'train':
            data_path = data_path / 'si284'
        elif stage == 'valid':
            data_path = data_path / 'dev93'
            
        with open(data_path/'mixinfo_noise.json', "r") as f:
            mixinfo = list(json.load(f).values())
        for m in mixinfo:
            mix_fn = Path(m["wav_mixed_noise_reverb"])
            mix_fn = Path("").joinpath(*mix_fn.parts[-3:])
            mix_fn = data_path / mix_fn
            dataset.append(mix_fn)
            mic_position.append(m["rir_info_microphone_position"])

            if return_ref:
                #ref_fns_list = m["wav_dpath_image_reverberant"]
                if self.ref_is_reverb:
                    ref_fns_list = m["wav_dpath_image_reverberant"]
                else:
                    ref_fns_list = m["wav_dpath_image_anechoic"]
                ref_fns = [Path(p) for p in ref_fns_list]
                ref_fns = [data_path / (Path("").joinpath(*fn.parts[-3:])) for fn in ref_fns]
                ref.append(ref_fns)

        if return_ref:
            return dataset, mic_position, ref
        else:
            return dataset, mic_position

    def __getitem__(self, idx):

        if not self.eval_with_wsj:
            mix_fn = self.all_wavs[idx]

            audio_mix, fs = torchaudio.load(mix_fn['wav_path'])
            audio_mix = audio_mix / audio_mix.max() * 0.95

            doa = torch.load(mix_fn['doa_path'])

            # now we know the number of channels
            n_channels = audio_mix.shape[0]

            # randomly shuffle the order of the channels in the mixture if required
            if self.shuffle_channels:
                p = torch.randperm(n_channels)
            else:
                p = torch.arange(n_channels)

            mic_position = torch.tensor(self.get_mic_position())

            audio_mix = audio_mix[p]
            mic_position = mic_position[p]

            if self.max_len_s is None:
                return audio_mix, mic_position

            else:
                n_target = int(fs * self.max_len_s)
                if audio_mix.shape[-1] < n_target:
                    offset = n_target - audio_mix.shape[-1]
                    s = random.randint(0, offset)
                    zeros = torch.zeros((audio_mix.shape[0], offset))
                    audio_mix = torch.cat((zeros[...,:s], audio_mix, zeros[..., s:]), dim=-1)
                else:
                    s = random.randint(0,audio_mix.shape[-1]-n_target)
                    e = s + n_target
                    audio_mix = audio_mix[..., s:e]

                return audio_mix, mic_position, doa, doa.shape[0]

        # validate with wsj dataset
        else:
            mix_fn = self.wsj_mix[idx]
            ref_fn = self.wsj_ref[idx]

            audio_mix, fs = torchaudio.load(mix_fn)

            audio_ref1, _ = torchaudio.load(ref_fn[0])
            audio_ref2, _ = torchaudio.load(ref_fn[1])

            #audio_ref1 /= audio_mix.max() * 0.95
            #audio_ref2 /= audio_mix.max() * 0.95
            #audio_mix /= audio_mix.max() * 0.95

            # now we know the number of channels
            n_channels = audio_mix.shape[0]

            audio_ref1, audio_ref2 =audio_ref1[0], audio_ref2[0]
            audio_ref = torch.stack((audio_ref1, audio_ref2),dim=0)
            
            mic_position = torch.tensor(self.wsj_mic_positions[idx])

            return audio_mix, audio_ref, mic_position



def collator(batch_list):
    """
    Collate a bunch of multichannel signals based
    on the size of the shortest sample. The samples are cut at the center
    """

    max_len = max([s[0].shape[-1] for s in batch_list])
    n_channels = batch_list[0][0].shape[0]

    batch_size = (len(batch_list), n_channels, max_len)

    data = batch_list[0][0].new_zeros(batch_size)
    target = batch_list[0][1].new_zeros(batch_size)

    offsets = [(max_len - s[0].shape[-1]) // 2 for s in batch_list]

    for b, ((d, t, mc, mp), o) in enumerate(zip(batch_list, offsets)):
        data[b, :, o : o + d.shape[-1]] = d
        target[b, :, o : o + t.shape[-1]] = t

    return data, target


def collator_6ch(batch_list):
    """
    Collate a bunch of multichannel signals based
    on the size of the shortest sample. The samples are cut at the center
    """


    max_len = max([s[0].shape[-1] for s in batch_list])
    mix_n_channels = batch_list[0][0].shape[0]
    ref_n_channels = batch_list[0][1].shape[0]

    mix_batch_size = (len(batch_list), mix_n_channels, max_len)
    ref_batch_size = (len(batch_list), ref_n_channels, max_len)
    mc_batch_size = (len(batch_list), 3)
    mp_batch_size = (len(batch_list), mix_n_channels, 3)
    sp_batch_size = (len(batch_list), 2, 3)

    data = batch_list[0][0].new_zeros(mix_batch_size)
    target = batch_list[0][1].new_zeros(ref_batch_size)
    mic_center = torch.tensor(batch_list[0][2]).new_zeros(mc_batch_size)
    mic_position = batch_list[0][3].new_zeros(mp_batch_size)
    speaker_position = torch.tensor(batch_list[0][4]).new_zeros(sp_batch_size)

    offsets = [(max_len - s[0].shape[-1]) // 2 for s in batch_list]

    for b, ((d, t, mc, mp, sp), o) in enumerate(zip(batch_list, offsets)):
        data[b, :, o : o + d.shape[-1]] = d
        target[b, :, o : o + t.shape[-1]] = t
        mic_center[b] = torch.tensor(mc)
        mic_position[b] = mp
        speaker_position[b] = torch.tensor(sp)

    return data, target, mic_center, mic_position, speaker_position


def collator_libri(batch_list):
    """
    Collate a bunch of multichannel signals based
    on the size of the shortest sample. The samples are cut at the center
    """

    max_len = max([s[0].shape[-1] for s in batch_list])
    mix_n_channels = batch_list[0][0].shape[0]

    mix_batch_size = (len(batch_list), mix_n_channels, max_len)
    mp_batch_size = (len(batch_list), mix_n_channels, 3)

    data = batch_list[0][0].new_zeros(mix_batch_size)
    mic_position = batch_list[0][1].new_zeros(mp_batch_size)

    offsets = [(max_len - s[0].shape[-1]) // 2 for s in batch_list]

    for b, ((d, mp), o) in enumerate(zip(batch_list, offsets)):
        data[b, :, o : o + d.shape[-1]] = d
        mic_position[b] = mp

    return data, mic_position


def collator_libri_doa(batch_list):
    """
    Collate a bunch of multichannel signals based
    on the size of the shortest sample. The samples are cut at the center
    """

    n_1src, n_2src = 0, 0
    for b in range(len(batch_list)):
        if batch_list[b][-1]==1:
            n_1src += 1
        elif batch_list[b][-1]==2:
            n_2src += 1


    max_len = max([s[0].shape[-1] for s in batch_list])
    mix_n_channels = batch_list[0][0].shape[0]

    mix_1src_batch_size = (n_1src, mix_n_channels, max_len)
    mix_2src_batch_size = (n_2src, mix_n_channels, max_len)

    mp_1src_batch_size = (n_1src, mix_n_channels, 3)
    mp_2src_batch_size = (n_2src, mix_n_channels, 3)

    doa_1src_batch_size = (n_1src, 1, 3)
    doa_2src_batch_size = (n_2src, 2, 3)


    mix_1src = batch_list[0][0].new_zeros(mix_1src_batch_size)
    mix_2src = batch_list[0][0].new_zeros(mix_2src_batch_size)
    mic_position_1src = batch_list[0][1].new_zeros(mp_1src_batch_size)
    mic_position_2src = batch_list[0][1].new_zeros(mp_2src_batch_size)
    doa_1src = batch_list[0][2].new_zeros(doa_1src_batch_size)
    doa_2src = batch_list[0][2].new_zeros(doa_2src_batch_size)

    offsets = [(max_len - s[0].shape[-1]) // 2 for s in batch_list]

    b_1src, b_2src = 0, 0
    for b, ((d, mp, doa, n_src), o) in enumerate(zip(batch_list, offsets)):
        if n_src == 1:
            mix_1src[b_1src, :, o : o + d.shape[-1]] = d
            mic_position_1src[b_1src] = mp
            doa_1src[b_1src] = doa
            b_1src += 1
        elif n_src == 2:
            mix_2src[b_2src, :, o : o + d.shape[-1]] = d
            mic_position_2src[b_2src] = mp
            doa_2src[b_2src] = doa
            b_2src += 1


    return mix_1src, mic_position_1src, doa_1src, mix_2src, mic_position_2src, doa_2src


def collator_6ch_2(batch_list):
    """
    Collate a bunch of multichannel signals based
    on the size of the shortest sample. The samples are cut at the center
    """


    max_len = max([s[0].shape[-1] for s in batch_list])
    mix_n_channels = batch_list[0][0].shape[0]
    ref_n_channels = batch_list[0][1].shape[0]

    mix_batch_size = (len(batch_list), mix_n_channels, max_len)
    ref_batch_size = (len(batch_list), ref_n_channels, max_len)
    mp_batch_size = (len(batch_list), mix_n_channels, 3)

    data = batch_list[0][0].new_zeros(mix_batch_size)
    target = batch_list[0][1].new_zeros(ref_batch_size)
    mic_position = batch_list[0][2].new_zeros(mp_batch_size)

    offsets = [(max_len - s[0].shape[-1]) // 2 for s in batch_list]

    for b, ((d, t, mp), o) in enumerate(zip(batch_list, offsets)):
        data[b, :, o : o + d.shape[-1]] = d
        target[b, :, o : o + t.shape[-1]] = t
        mic_position[b] = mp

    return data, target, mic_position 


class InterleavedLoaders:
    """
    This is a wrapper to sample alternatively from several dataloaders

    It samples until all of them are exhausted

    Parameters
    ----------
    dataloaders: list of torch.utils.data.DataLoader
        A list that contains all the dataloaders we want to sample from
    """

    def __init__(self, dataloaders: List[torch.utils.data.DataLoader]):
        self.dataloaders = dataloaders
        self._reset_dataloader_iter()

    def _reset_dataloader_iter(self):
        self.dataloaders_iter = [iter(d) for d in self.dataloaders]

    def __len__(self):
        return sum([len(dl) for dl in self.dataloaders])

    def __iter__(self):
        return self

    def __next__(self):
        for i, dl in enumerate(self.dataloaders_iter):
            try:
                # we loop until we find a non-empty loader
                batch = next(dl)
                # then we put dataloader at the end of the list
                self.dataloaders_iter = (
                    self.dataloaders_iter[i + 1 :] + self.dataloaders_iter[: i + 1]
                )
                # return the batch
                return batch
            except StopIteration:
                continue

        self._reset_dataloader_iter()

        raise StopIteration
