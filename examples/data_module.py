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

import json
from pathlib import Path

import pytorch_lightning as pl
import torch
import torchaudio
from torch.utils.data import DataLoader

from dataloader import (
    InterleavedLoaders,
    WSJ1SpatialDataset,
    collator,
    collator_6ch,
)


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
        ref_is_reverb=None,
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
            )

        self.wsj_val = [
            WSJ1SpatialDataset(
                self.dataset_location / "dev93",
                max_len_s=None,
                shuffle_channels=False,
                shuffle_ref=False,
                noiseless=self.noiseless,
                ref_is_reverb=self.ref_is_reverb,
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
                batch_size=self.batch_size,
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
                # pin_memory=True,
            )
            for ds in self.wsj_test
        ]
        return wsj_test
