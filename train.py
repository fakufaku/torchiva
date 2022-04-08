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
from pathlib import Path

import pytorch_lightning as pl
import torch
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint


from data_module import WSJDataModule, WSJ6chDataModule, LibriCSSDataModule
from separation_model import SeparationModel, UnsupervisedDOAModel, UnsupervisedLibriModel

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Trains a model for AuxIVA")
    parser.add_argument("config", type=Path, help="The configuration file")
    parser.add_argument(
        "dataset", type=Path, help="Location of the dataset metadata file"
    )
    parser.add_argument("--resume", type=str, help="Location of checkpoint to load")
    parser.add_argument(
        "--workers", type=int, default=8, help="Number of workers for the dataloader"
    )
    parser.add_argument(
        "--gpu", type=int, default=0, help="GPU number."
    )
    args = parser.parse_args()

    with open(args.config, "r") as f:
        config = json.load(f)

    # seed all RNGs for deterministic behavior
    pl.seed_everything(config["training"]["seed"])

    # root dir
    root_dir = os.path.join(config["checkpoints"], config["name"])

    # some bookkeeping for torch and torchaudio
    torch.autograd.set_detect_anomaly(True)

    # configure checkpointing to save all models
    checkpoint_callback = pl.callbacks.ModelCheckpoint(
        monitor="val_loss", save_top_k=-1, mode="min"
    )

    try:
        return_mic_position=config["training"]["return_mic_position"]
    except KeyError:
        return_mic_position=False

    # load the dataset
    train_n_batch_sizes = (
        config["training"]["n_batch_sizes"]
        if "n_batch_sizes" in config["training"]
        else None
    )
    valid_n_channels = (
        config["training"]["valid_n_channels"]
        if "valid_n_channels" in config["training"]
        else None
    )

    
    if "MVDR" in config["model"]["source_model"]["name"]:
        model = MaskBasedMVDRBeamformingModelWSJ(config)
    else:
        model = UnsupervisedDOAModel(config)

    dm = WSJ6chDataModule(
        batch_size=config["training"]["batch_size"],
        shuffle=config["training"]["shuffle"],
        num_workers=args.workers,
        dataset_location=args.dataset,
        shuffle_ref=config["training"]["shuffle_ref"],
        shuffle_channels = config["training"]["shuffle_channels"],
        train_n_channels=config["training"]["n_channels"],
        train_n_batch_sizes=train_n_batch_sizes,
        valid_n_channels=valid_n_channels,
        max_len_s=config["training"]["max_len_s"],
        ref_is_reverb=config["training"]["ref_is_reverb"],
        noiseless=config["training"]["noiseless"],
        return_mic_position=return_mic_position,
    )
    
    # create a logger
    os.makedirs(root_dir, exist_ok=True)
    tb_logger = pl_loggers.TensorBoardLogger(root_dir)
    
    # most basic trainer, uses good defaults (auto-tensorboard, checkpoints, logs, and more)

    if "max_epoch" in config["training"]:
        max_epoch = config["training"]["max_epoch"]
    else:
        max_epoch = 100
    
    trainer = pl.Trainer(
        deterministic=deterministic,
        gpus=[args.gpu],
        check_val_every_n_epoch=1,
        default_root_dir=root_dir,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=args.resume,
        logger=tb_logger,
        profiler="simple",
        max_epochs = max_epoch,
    )
    trainer.fit(model, dm)