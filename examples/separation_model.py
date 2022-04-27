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


import itertools
import bisect
import pytorch_lightning as pl
import torch

import torchiva as bss

import source_models
import fast_bss_eval
from autoclip_module import AutoClipper

import random


def _get_grad_norm(model):
    """
    Helper function that computes the gradient norm
    """
    total_norm = 0
    for p in model.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** (1.0 / 2)
    return total_norm



class WSJModel(pl.LightningModule):
    def __init__(self, config):
        # init superclass
        super().__init__()

        random.seed(config["training"]["seed"])

        # save all variables in __init__ signature to self.hparams
        self.save_hyperparameters()

        # Training parameters
        self.lr = config["training"]["lr"]
        self.weight_decay = config["training"]["weight_decay"]
        self.lr_scheduler_kwargs = config["training"]["lr_scheduler_kwargs"]

        self.loss_type = config["training"]["loss_type"]
        self.algo=config['algorithm']

        # auto clip
        try:
            self.autoclipper = AutoClipper(config["training"]["autoclip_p"])
        except KeyError:
            self.autoclipper = AutoClipper(100)
        
        self.n_iter = config["model"]["n_iter"]
        self.n_chan = config["training"]["n_chan"]
        self.n_src = config["training"]["n_src"]
        self.n_fft = config["model"]["n_fft"]
        self.hop_length = config["model"]["hop_length"]

        try:
            n_power_iter = config["model"]["n_power_iter"]
        except KeyError:
            n_power_iter = None
        
        try:
            source_model = source_models.get_model(**config["model"]["source_model"])
        except KeyError:
            source_model = None

        self.separator = bss.nn.BSSSeparator(
            self.n_fft,
            self.n_iter,
            hop_length=self.hop_length,
            n_taps=config["model"]["n_taps"],
            n_delay=config["model"]["n_delay"],
            n_src=self.n_src,
            algo=self.algo,
            source_model=source_model, 
            proj_back_mic=config["model"]["ref_mic"],
            use_dmc=config["model"]["use_dmc"],
            n_power_iter=n_power_iter,
        )


    def forward(self, x):
        y_hat = self.separator(x)
        return y_hat


    def compute_metrics(self, y_hat, y):

        metrics = {}
        m = min([y_hat.shape[-1], y.shape[-1]])
        y_hat = y_hat[...,:m]
        y = y[...,:m]

        y_hat = bss.select_most_energetic(
            y_hat, num=self.n_src, dim=-2, dim_reduc=-1,
        )

        metrics['cisdr_loss'] = fast_bss_eval.sdr_pit_loss(y_hat, y, clamp_db=50).mean()

        return metrics


    def training_step(self, batch, batch_idx):

        x, y = batch
        y_hat = self(x[..., :self.n_chan, :])

        metrics = self.compute_metrics(y_hat, y)

        loss = metrics['cisdr_loss']

        cur_step = self.trainer.global_step
        if cur_step % 5 == 0:
            self.logger.log_metrics(metrics, step=cur_step)
        
        return loss


    def on_validation_epoch_start(self):
        self.datasets_types = set()
        self.val_metrics = {}

    def validation_step(self, batch, batch_idx, dataset_i=0):

        with torch.no_grad():
            x, y = batch

            y_hat = self(x[..., :self.n_chan, :])

            metrics = self.compute_metrics(y_hat, y)
            metrics['val_loss'] = metrics['cisdr_loss']

        return metrics


    def validation_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """

        if not isinstance(outputs, list):
            outputs = [outputs]

        avg_loss = {
            'cisdr': 0,
            'val_loss':0,
        }

        for results in outputs:
            avg_loss['cisdr'] -= results['cisdr_loss']
            avg_loss['val_loss'] += results['val_loss']
        
        avg_loss['cisdr'] /= len(outputs)
        avg_loss['val_loss'] /= len(outputs)

        self.log("val_loss", avg_loss['val_loss'])

        for loss_type, loss_value in avg_loss.items():
            if loss_type != 'val_loss':
                self.log(f"val_{loss_type}", loss_value)

    '''
    def on_test_epoch_start(self):
        self.on_validation_epoch_start()

    def test_step(self, batch, batch_idx, dataset_i=0):
        return self.validation_step(batch, batch_idx, dataset_i=0)

    def test_epoch_end(self, outputs):
        """
        Called at the end of validation to aggregate outputs.
        :param outputs: list of individual outputs of each validation step.
        """
        self.validation_epoch_end()
    '''
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                **self.lr_scheduler_kwargs,
            ),
            "monitor": f"val_loss",
        }

    def on_after_backward(self):
        grad_norm, clipping_threshold = self.autoclipper(self)

        # log every few iterations
        if self.trainer.global_step % 25 == 0:
            clipped_norm = min(grad_norm, clipping_threshold)

            # get the current learning reate
            opt = self.trainer.optimizers[0]
            current_lr = opt.state_dict()["param_groups"][0]["lr"]

            self.logger.log_metrics(
                {
                    "grad/norm": grad_norm,
                    "grad/clipped_norm": clipped_norm,
                    "grad/step_size": current_lr * clipped_norm,
                },
                step=self.trainer.global_step,
            )