# Copyright 2021 Robin Scheibler
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
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchaudio
from torch.utils.data import DataLoader

from dataloader import (
    InterleavedLoaders,
    BSSSpeechDataset,
    WSJ1SpatialDataset,
    LibriCSSDataset,
    LibriCSSDataset_DOA,
    collator,
    collator_6ch,
    collator_libri,
    collator_libri_doa,
    collator_6ch_2,
)


class WSJDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=5,
        shuffle=True,
        num_workers=0,
        dataset_location="",
        max_len_s=None,
        shuffle_channels=True,
        shuffle_ref=False,
        train_n_channels=2,
        train_n_batch_sizes=None,
        valid_n_channels=None,
        noiseless=False,
        ref_is_reverb = None,
        return_mic_position=False,
    ):
        super().__init__()

        # set regular parameters
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.dataset_location = dataset_location

        if max_len_s == -1:
            max_len_s = None
        self.max_len_s = max_len_s
        self.noiseless = noiseless
        self.ref_is_reverb = ref_is_reverb

        self.shuffle_channels = shuffle_channels
        self.shuffle_ref = shuffle_ref

        self.return_mic_position = return_mic_position

        # handle the number of samples per number of channels
        if isinstance(train_n_channels, list):
            train_n_channels = dict(
                zip(train_n_channels, [None] * len(train_n_channels))
            )
        elif not isinstance(train_n_channels, dict):
            train_n_channels = {train_n_channels: None}

        self.train_n_channels = {}
        for nch, nsamples in train_n_channels.items():
            if nsamples == -1:
                self.train_n_channels[int(nch)] = None
            else:
                self.train_n_channels[int(nch)] = nsamples

        # handles different batch sizes for different number of channels
        self.train_n_batch_sizes = {}
        if train_n_batch_sizes is None:
            for nch in self.train_n_channels:
                self.train_n_batch_sizes[nch] = batch_size
        else:
            for nch in self.train_n_channels:
                self.train_n_batch_sizes[nch] = train_n_batch_sizes[str(nch)]

        # check that everything is provided
        self.available_n_channels = [2, 3, 4]
        for tch in self.train_n_channels:
            assert (
                tch in self.available_n_channels
            ), "Only 2, 3, 4 channels available (number of samples)"
        for tch in self.train_n_batch_sizes:
            assert (
                tch in self.available_n_channels
            ), "Only 2, 3, 4 channels available (batch sizes)"

        # what do we use for validation ?
        if valid_n_channels is None:
            self.valid_n_channels = self.available_n_channels
        else:
            assert isinstance(
                valid_n_channels, list
            ), "the validation channels should be in a list"
            self.valid_n_channels = valid_n_channels

    def setup(self, stage):
        #torchaudio.set_audio_backend("sox")
        torchaudio.set_audio_backend("sox_io")

        self.wsj_train = {}
        for m, n_samples in self.train_n_channels.items():
            self.wsj_train[m] = WSJ1SpatialDataset(
                self.dataset_location / f"wsj1_{m}_mix_m{m}/si284",
                max_len_s=self.max_len_s,
                shuffle_channels=self.shuffle_channels,
                shuffle_ref=self.shuffle_ref,
                noiseless=self.noiseless,
                ref_is_reverb=self.ref_is_reverb,
                max_n_samples=n_samples,
                return_mic_position=self.return_mic_position,
            )

        self.wsj_val = [
            WSJ1SpatialDataset(
                self.dataset_location / f"wsj1_{m}_mix_m{m}/dev93",
                max_len_s=None,
                shuffle_channels=False,
                shuffle_ref=False,
                noiseless=self.noiseless,
                ref_is_reverb=self.ref_is_reverb,
                return_mic_position=self.return_mic_position,
            )
            for m in self.valid_n_channels
        ]
        self.wsj_test = [
            WSJ1SpatialDataset(
                self.dataset_location / f"wsj1_{m}_mix_m{m}/eval92",
                max_len_s=None,
                shuffle_channels=False,
                shuffle_ref=False,
                noiseless=self.noiseless,
                ref_is_reverb=self.ref_is_reverb,
                return_mic_position=self.return_mic_position,
            )
            for m in self.valid_n_channels
        ]

    def train_dataloader(self):
        wsj_train = InterleavedLoaders(
            [
                DataLoader(
                    ds,
                    batch_size=self.train_n_batch_sizes[m],
                    shuffle=self.shuffle,
                    num_workers=self.num_workers,
                    collate_fn=collator,
                    pin_memory=True,
                )
                for m, ds in self.wsj_train.items()
            ]
        )
        return wsj_train

    def val_dataloader(self):
        wsj_val = [
            DataLoader(
                ds,
                batch_size=4,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collator,
                pin_memory=True,
            )
            for ds in self.wsj_val
        ]
        return wsj_val

    def test_dataloader(self):
        wsj_test = [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collator,
                #pin_memory=True,
            )
            for ds in self.wsj_test
        ]
        return wsj_test


class WSJ6chDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=5,
        shuffle=True,
        num_workers=0,
        dataset_location="",
        max_len_s=None,
        shuffle_channels=True,
        shuffle_ref=False,
        train_n_channels=2,
        train_n_batch_sizes=None,
        valid_n_channels=None,
        noiseless=False,
        ref_is_reverb = None,
        return_mic_position=True,
    ):
        super().__init__()

        # set regular parameters
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.dataset_location = dataset_location

        if max_len_s == -1:
            max_len_s = None
        self.max_len_s = max_len_s
        self.noiseless = noiseless
        self.ref_is_reverb = ref_is_reverb

        self.shuffle_channels = shuffle_channels
        self.shuffle_ref = shuffle_ref

        self.return_mic_position = return_mic_position

        
        # handle the number of samples per number of channels
        if isinstance(train_n_channels, list):
            train_n_channels = dict(
                zip(train_n_channels, [None] * len(train_n_channels))
            )
        elif not isinstance(train_n_channels, dict):
            train_n_channels = {train_n_channels: None}

        self.train_n_channels = {}
        for nch, nsamples in train_n_channels.items():
            if nsamples == -1:
                self.train_n_channels[int(nch)] = None
            else:
                self.train_n_channels[int(nch)] = nsamples

        # handles different batch sizes for different number of channels
        self.train_n_batch_sizes = {}
        if train_n_batch_sizes is None:
            for nch in self.train_n_channels:
                self.train_n_batch_sizes[nch] = batch_size
        else:
            for nch in self.train_n_channels:
                self.train_n_batch_sizes[nch] = train_n_batch_sizes[str(nch)]
        

    def setup(self, stage):
        torchaudio.set_audio_backend("sox_io")

        self.wsj_train = {}
        for m, n_samples in self.train_n_channels.items():
            self.wsj_train[m] = WSJ1SpatialDataset(
                self.dataset_location / "si284",
                max_len_s=self.max_len_s,
                shuffle_channels=self.shuffle_channels,
                shuffle_ref=self.shuffle_ref,
                noiseless=self.noiseless,
                ref_is_reverb=self.ref_is_reverb,
                max_n_samples=n_samples,
                return_mic_position=self.return_mic_position,
            )

        self.wsj_val = [
            WSJ1SpatialDataset(
                self.dataset_location / "dev93",
                max_len_s=None,
                shuffle_channels=False,
                shuffle_ref=False,
                noiseless=self.noiseless,
                ref_is_reverb=self.ref_is_reverb,
                return_mic_position=self.return_mic_position,
            )
        ]
        self.wsj_test = [
            WSJ1SpatialDataset(
                self.dataset_location / "eval92",
                max_len_s=None,
                shuffle_channels=False,
                shuffle_ref=False,
                noiseless=self.noiseless,
                ref_is_reverb=self.ref_is_reverb,
                return_mic_position=self.return_mic_position,
            )
        ]

    def train_dataloader(self):
        wsj_train = InterleavedLoaders(
            [
                DataLoader(
                    ds,
                    batch_size=self.train_n_batch_sizes[m],
                    shuffle=self.shuffle,
                    num_workers=self.num_workers,
                    collate_fn=collator_6ch,
                    pin_memory=True,
                )
                for m, ds in self.wsj_train.items()
            ]
        )
        return wsj_train

    def val_dataloader(self):
        wsj_val = [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collator_6ch,
                pin_memory=True,
            )
            for ds in self.wsj_val
        ]
        return wsj_val

    def test_dataloader(self):
        wsj_test = [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collator_6ch,
                #pin_memory=True,
            )
            for ds in self.wsj_test
        ]
        return wsj_test


class LibriCSSDataModule(pl.LightningDataModule):
    def __init__(
        self,
        batch_size=5,
        shuffle=True,
        num_workers=0,
        dataset_location="",
        max_len_s=None,
        shuffle_channels=True,
        eval_with_wsj=False,
        ref_is_reverb=True,
    ):
        super().__init__()

        # set regular parameters
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.dataset_location = dataset_location
        self.eval_with_wsj=eval_with_wsj
        self.ref_is_reverb = ref_is_reverb

        if max_len_s == -1:
            max_len_s = None
        self.max_len_s = max_len_s

        self.shuffle_channels = shuffle_channels


    def setup(self, stage):
        torchaudio.set_audio_backend("sox_io")
        '''
        self.wsj_train = [
            LibriCSSDataset(
                self.dataset_location,
                max_len_s=self.max_len_s,
                shuffle_channels=self.shuffle_channels,
                stage='train',
            )
        ]
        
        max_len_s = None
        self.wsj_val = [
            LibriCSSDataset(
                self.dataset_location,
                max_len_s=max_len_s,
                shuffle_channels=False,
                stage='valid',
                eval_with_wsj=self.eval_with_wsj,
            )
        ]
        '''

        self.wsj_train = [
            LibriCSSDataset_DOA(
                self.dataset_location,
                max_len_s=self.max_len_s,
                shuffle_channels=self.shuffle_channels,
                stage='train',
            )
        ]
        
        max_len_s = None
        self.wsj_val = [
            LibriCSSDataset_DOA(
                self.dataset_location,
                max_len_s=max_len_s,
                shuffle_channels=False,
                stage='valid',
                eval_with_wsj=self.eval_with_wsj,
                ref_is_reverb=self.ref_is_reverb,
            )
        ]


    def train_dataloader(self):
        wsj_train = InterleavedLoaders(
            [
                DataLoader(
                    ds,
                    batch_size=self.batch_size,
                    shuffle=self.shuffle,
                    num_workers=self.num_workers,
                    #collate_fn=collator_libri,
                    collate_fn=collator_libri_doa,
                    pin_memory=True,
                )
                for ds in self.wsj_train
            ]
        )
        return wsj_train
    
    def val_dataloader(self):
        if self.eval_with_wsj:
            collate_fn=collator_6ch_2
        else:
            #collate_fn=collator_libri
            collate_fn=collator_libri_doa
        wsj_val = [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=False,
                num_workers=self.num_workers,
                collate_fn=collate_fn,
                pin_memory=True,
            )
            for ds in self.wsj_val
        ]
        return wsj_val



class CMUDataModule(pl.LightningDataModule):
    def __init__(
        self,
        dataset_metadata,
        batch_size=5,
        shuffle=True,
        num_workers=0,
        max_len_s=None,
        shuffle_channels=True,
    ):
        super().__init__()
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.num_workers = num_workers
        self.max_len_s = max_len_s
        self.shuffle_channels = shuffle_channels

        dataset_metadata = Path(dataset_metadata)
        self.data_root = dataset_metadata.parent
        with open(dataset_metadata, "r") as f:
            self.metadata = json.load(f)

        self.n_train = int(0.7 * len(self.metadata["2_channels"]))

    def setup(self, stage):
        torchaudio.set_audio_backend("sox")

        self.cmu_train = BSSSpeechDataset(
            self.data_root, self.metadata["2_channels"][: self.n_train]
        )
        self.cmu_val = [
            BSSSpeechDataset(self.data_root, self.metadata[key][self.n_train :])
            for key in self.metadata.keys()
        ]

    def train_dataloader(self):
        cmu_train = DataLoader(
            self.cmu_train,
            batch_size=self.batch_size,
            shuffle=self.shuffle,
            num_workers=self.num_workers,
            collate_fn=collator,
        )
        return cmu_train

    def val_dataloader(self):
        cmu_val = [
            DataLoader(
                ds,
                batch_size=self.batch_size,
                shuffle=self.shuffle,
                num_workers=self.num_workers,
                collate_fn=collator,
            )
            for ds in self.cmu_val
        ]
        return cmu_val[:-1]  # remove the 8 channels dataset
