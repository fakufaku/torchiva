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


import itertools
import bisect
import pytorch_lightning as pl
import torch

import torchiva as bss
from doa_loss import doa_loss, doa_loss_low_overlap, doa_loss2
from doa_loss import est_number_of_sources_via_doa

import source_models
import mask_models

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



class UnsupervisedDOAModel(pl.LightningModule):
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

        self.proj_back = config["model"]["proj_back"]

        # auto clip
        try:
            self.autoclipper = AutoClipper(config["training"]["autoclip_p"])
        except KeyError:
            self.autoclipper = AutoClipper(100)
        
        self.n_iter = config["model"]["n_iter"]


        self.n_fft = config["model"]["n_fft"]
        self.hop_length = config["model"]["hop_length"]
        
        try:
            source_model = source_models.get_model(**config["model"]["source_model"])
        except KeyError:
            source_model = None

        try:
            bss_source_model = bss.models.get_model(config["model"]["bss_source_model"])
        except KeyError:
            bss_source_model = bss.models.GaussModel()
        

        try:
            self.supervised = config["training"]["supervised"]
        except KeyError:
            self.supervised = False

        try:
            bss_n_iter = config["model"]["bss_n_iter"]
        except KeyError:
            bss_n_iter = 0

        try:
            self.n_grid = config["model"]["n_grid"]
        except KeyError:
            self.n_grid = 300

        self.doa_ratio = config["training"]["doa_ratio"]
        self.kld_ratio = config["training"]["kld_ratio"]
        self.ivasdr_ratio = config["training"]["ivasdr_ratio"]

        self.dnn_nchan = config["training"]["dnn_nchan"]
        self.bss_nchan = config["training"]["bss_nchan"]

        self.doa_nsrc = config["training"]["doa_nsrc"]
        self.kld_nsrc = config["training"]["kld_nsrc"]
        self.n_src = max(self.doa_nsrc, self.kld_nsrc)
        
        try:
            self.spherical = config["training"]["spherical"]
        except KeyError:
            self.spherical = True

        try:
            self.wpe_n_fft = config["model"]["wpe_n_fft"]
            self.wpe_n_iter = config["model"]["wpe_n_iter"]
            self.wpe_n_taps = config["model"]["wpe_n_taps"]
            self.wpe_n_delay = config["model"]["wpe_n_delay"]
        except KeyError:
            self.wpe_n_fft = 512
            self.wpe_n_iter = 3
            self.wpe_n_taps = 10
            self.wpe_n_delay = 3

        self.separator = bss.nn.Separator(
            self.n_fft,
            self.n_iter,
            hop_length = self.hop_length,
            n_taps=config["model"]["n_taps"],
            n_delay=config["model"]["n_delay"],
            n_src=self.n_src,
            dnn_nchan=self.dnn_nchan,
            bss_nchan=self.bss_nchan,
            algo=config["algorithm"],
            source_model=source_model,
            bss_source_model=bss_source_model,
            bss_n_iter=bss_n_iter,
            wpe_n_fft = self.wpe_n_fft,
            wpe_n_iter = self.wpe_n_iter,
            wpe_n_taps = self.wpe_n_taps,
            wpe_n_delay = self.wpe_n_delay, 
            proj_back_mic=config["model"]["ref_mic"],
        )

        self.stft = bss.STFT(self.n_fft, hop_length=self.hop_length)


    def forward(self, x, n_iter=None):
        if n_iter is None:
            n_iter = self.n_iter

        #x = self.separator(x, n_iter=n_iter)
        Y_hat, Y_bss, weight, weight_bss, W, _ = self.separator(
            x, n_iter=n_iter,
        )
        return Y_hat, Y_bss, weight, weight_bss, W


    def compute_metrics(self, y_hat, y):

        metrics = {}
        m = min([y_hat.shape[-1], y.shape[-1]])
        y_hat = y_hat[...,:m]
        y = y[...,:m]

        y_hat = bss.select_most_energetic(
            y_hat, num=2, dim=-2, dim_reduc=-1,
        )

        metrics['cisdr_loss'] = fast_bss_eval.sdr_pit_loss(y_hat, y, clamp_db=50).mean()

        return metrics


    def training_step(self, batch, batch_idx):

        x, y, mic_center, mic_position, speaker_position = batch
        Y_hat, Y_bss, weight, weight_bss, W = self(x)

        y_hat = self.stft.inv(Y_hat)
        y_bss = self.stft.inv(Y_bss)

        metrics = self.compute_metrics(y_hat, y)

        loss = 0

        if 'kld' in self.loss_type:
            metrics['kld_loss'] = bss.KLDLoss(
                Y_hat, 
                Y_bss, 
                weight, 
                weight_bss, 
                clamp_thres=50,
                spherical=self.spherical,
            )
            loss += metrics['kld_loss'] * self.kld_ratio

        if 'iva' in self.loss_type:
            iva_cisdr_loss = fast_bss_eval.sdr_pit_loss(y_hat, y_bss, clamp_db=25).mean()
            metrics['iva_cisdr_loss'] = iva_cisdr_loss
            loss += metrics['iva_cisdr_loss'] * self.ivasdr_ratio

        if 'doa' in self.loss_type:
            _, _, W = bss.most_energetic_in_freq(
                Y_hat,  weight, W, num=self.doa_nsrc, dim=-3, dim_reduc=(-1,-2),
            )

            if self.supervised:
                # compute true doa
                idx_1src, idx_2src, doa_1src, doa_2src = est_number_of_sources_via_doa(
                    self.stft(x),
                    mic_position,
                    self.n_fft,
                    mic_center=mic_center,
                    speaker_position=speaker_position,
                    n_grid = self.n_grid,
                    n_iter=30,
                    z_only_positive=False,
                )

            else:
                # estimate doa
                idx_1src, idx_2src, doa_1src, doa_2src = est_number_of_sources_via_doa(
                    self.stft(x),
                    mic_position,
                    self.n_fft,
                    n_grid = self.n_grid,
                    n_iter=30,
                    z_only_positive=False,
                )

            doaloss = doa_loss2(
                doa_2src,
                W[idx_2src],
                mic_position[idx_2src],
                self.n_fft,
            )
    
            metrics['doa_loss'] = doaloss
            loss += metrics['doa_loss'] * self.doa_ratio

        if self.loss_type == 'cisdr':
            loss = metrics[self.loss_type+'_loss']

        cur_step = self.trainer.global_step
        if cur_step % 5 == 0:
            self.logger.log_metrics(metrics, step=cur_step)
        
        return loss


    def on_validation_epoch_start(self):
        self.datasets_types = set()
        self.val_metrics = {}

    def validation_step(self, batch, batch_idx, dataset_i=0):

        with torch.no_grad():
            x, y, mic_center, mic_position, speaker_position = batch

            Y_hat, Y_bss, weight, weight_bss, W = self(x)

            y_hat = self.stft.inv(Y_hat)
            y_bss = self.stft.inv(Y_bss)

            metrics = self.compute_metrics(y_hat, y)

            val_loss = 0 
            if self.loss_type == 'cisdr':
                val_loss = metrics['cisdr_loss']

            if 'kld' in self.loss_type:
                metrics['kld_loss'] = bss.KLDLoss(
                    Y_hat, 
                    Y_bss, 
                    weight, 
                    weight_bss, 
                    clamp_thres=50,
                    spherical=self.spherical,
                )
                val_loss += metrics['kld_loss'] * self.kld_ratio

            if 'iva' in self.loss_type:
                iva_cisdr_loss = fast_bss_eval.sdr_pit_loss(y_hat, y_bss, clamp_db=25).mean()
                metrics['iva_cisdr_loss'] = iva_cisdr_loss
                val_loss += iva_cisdr_loss

            if 'doa' in self.loss_type:

                _, _, W = bss.most_energetic_in_freq(
                    Y_hat,  weight, W, num=self.doa_nsrc, dim=-3, dim_reduc=(-1,-2),
                )

                if self.supervised:
                    # compute true doa
                    idx_1src, idx_2src, doa_1src, doa_2src = est_number_of_sources_via_doa(
                        self.stft(x),
                        mic_position,
                        self.n_fft,
                        mic_center=mic_center,
                        speaker_position=speaker_position,
                        n_grid = self.n_grid,
                        n_iter=30,
                        z_only_positive=False,
                    )

                else:
                    # estimate doa
                    idx_1src, idx_2src, doa_1src, doa_2src = est_number_of_sources_via_doa(
                        self.stft(x),
                        mic_position,
                        self.n_fft,
                        n_grid = self.n_grid,
                        n_iter=30,
                        z_only_positive=False,
                    )

                doaloss = doa_loss2(
                    doa_2src,
                    W[idx_2src],
                    mic_position[idx_2src],
                    self.n_fft,
                )
                metrics['doa_loss'] = doaloss
                val_loss += doaloss * self.doa_ratio

            metrics['val_loss'] = val_loss

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


class MaskBasedMVDRBeamformingModelWSJ(pl.LightningModule):
    def __init__(self, config):
        # init superclass
        super().__init__()
        # save all variables in __init__ signature to self.hparams
        self.save_hyperparameters()

        # Training parameters
        self.weight_decay = config["training"]["weight_decay"]
        self.lr = config["training"]["lr"]
        self.lr_scheduler_kwargs = config["training"]["lr_scheduler_kwargs"]

        try:
            self.autoclipper = AutoClipper(config["training"]["autoclip_p"])
        except KeyError:
            self.autoclipper = AutoClipper(100)

        self.n_fft = config["model"]["n_fft"]

        try:
            bss_n_iter = config["model"]["bss_n_iter"]
        except KeyError:
            bss_n_iter = 0

        # The mask model
        mask_model = source_models.get_model(**config["model"]["source_model"])

        if "wpe_source_model" in config["model"]:
            wpe_source_model = source_models.get_model(**config["model"]["wpe_source_model"])
        else:
            wpe_source_model = None

        n_iter = config["model"]["wpe_n_iter"]
        n_taps = config["model"]["wpe_n_taps"]
        n_delay = config["model"]["wpe_n_delay"]
        wpe_n_fft = config["model"]["wpe_n_fft"]
        
        self.n_grid = config["model"]["n_grid"]

        try:
            self.dnn_nchan = config["training"]["dnn_nchan"]
            self.bss_nchan = config["training"]["bss_nchan"]
        except KeyError:
            self.dnn_nchan = None
            self.bss_nchan = None

        try:
            self.doa_ratio = config["training"]["doa_ratio"]
        except KeyError:
            self.doa_ratio = 0
        try:
            self.kld_ratio = config["training"]["kld_ratio"]
        except KeyError:
            self.kld_ratio = 0
        try:
            self.ivasdr_ratio = config["training"]["ivasdr_ratio"]
        except KeyError:
            self.ivasdr_ratio = 0

        try:
            self.spherical = config["training"]["spherical"]
        except KeyError:
            self.spherical = True

        self.mvdr = bss.nn.MVDRBeamformer(
            n_fft=self.n_fft, 
            mask_model=mask_model,
            wpe_model=wpe_source_model,
            wpe_n_iter=n_iter,
            wpe_n_taps=n_taps,
            wpe_n_delay=n_delay,
            wpe_n_fft=wpe_n_fft,
            bss_n_iter=bss_n_iter,
            bss_nchan=self.bss_nchan,
            dnn_nchan=self.dnn_nchan,
        )
        

        if "wpe_source_model" in config["model"]:
            wpe_source_model = source_models.get_model(**config["model"]["wpe_source_model"])
        else:
            wpe_source_model = None

        self.loss_type = config["training"]["loss_type"]

        # the stft engine
        self.stft = bss.STFT(self.n_fft)  # default parameters for hop and window

        # We may use a pre-trained model if available
        if "pretrained-model" in config["model"]:
            pretrained = torch.load(config["model"]["pretrained-model"])
            self.mask_model.load_state_dict(pretrained["state_dict"])

    def forward(self, x):

        Y, W, Y_bss, W_bss, weight_bss = self.mvdr(x)

        return Y, W, Y_bss, W_bss, weight_bss

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
            #"lr_scheduler": torch.optim.lr_scheduler.StepLR(
            #    optimizer,
            #    **self.lr_scheduler_kwargs,
            #),
            "monitor": f"val_loss",
        }

        return [optimizer], [scheduler]


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

    def compute_metrics(self, y_hat, y):

        metrics = {}
        m = min([y_hat.shape[-1], y.shape[-1]])
        y_hat = y_hat[...,:m]
        y = y[...,:m]

        metrics['cisdr_loss'] = fast_bss_eval.sdr_pit_loss(y_hat, y, clamp_db=50).mean()

        return metrics

    def training_step(self, batch, batch_idx):
        x, y, mic_center, mic_position, speaker_position = batch

        Y_hat, W, Y_bss, W_bss, weight_bss = self(x)
        loss = 0

        y_hat = self.stft.inv(Y_hat)
        y_bss = self.stft.inv(Y_bss)
        metrics = self.compute_metrics(y_hat, y)


        if 'iva' in self.loss_type:
            #y_hat = self.stft.inv(Y_hat)
            #y_bss = self.stft.inv(Y_bss)

            iva_cisdr_loss = fast_bss_eval.sdr_pit_loss(y_hat, y_bss, clamp_db=25).mean()
            metrics['iva_cisdr_loss'] = iva_cisdr_loss
            loss += metrics['iva_cisdr_loss'] * self.ivasdr_ratio

        if 'kld' in self.loss_type:
            weight = 1. / torch.clamp(Y_hat.real.square() + Y_hat.imag.square(), min=1e-5)

            metrics['kld_loss'] = bss.KLDLoss(
                Y_hat, 
                Y_bss, 
                weight, 
                weight_bss, 
                clamp_thres=50,
                spherical=self.spherical
            )
            loss += metrics['kld_loss'] * self.kld_ratio

        if 'doa' in self.loss_type:
            # estimate doa
            idx_1src, idx_2src, doa_1src, doa_2src = est_number_of_sources_via_doa(
                self.stft(x),
                #X,
                mic_position,
                self.n_fft,
                n_grid = self.n_grid,
                n_iter=30,
                z_only_positive=False,
            )
            doaloss = doa_loss2(
                doa_2src,
                W[idx_2src],
                mic_position[idx_2src],
                self.n_fft,
            )
    
            metrics['doa_loss'] = doaloss
            loss += metrics['doa_loss'] * self.doa_ratio
        
        if self.loss_type == 'cisdr':
            loss = metrics[self.loss_type+'_loss']
                
        cur_step = self.trainer.global_step
        if cur_step % 5 == 0:
            self.logger.log_metrics(metrics, step=cur_step)

        return loss


    def training_epoch_end(self, outputs):
        if not isinstance(outputs, list):
            outputs = [outputs]

        avg_loss = {
            'loss':0,
        }

        for results in outputs:
            avg_loss['loss'] += results['loss']

        avg_loss['loss'] /= len(outputs)

        for loss_type, loss_value in avg_loss.items():
            self.log(f"train_{loss_type}", loss_value)


    def validation_step(self, batch, batch_idx, dataset_i=0):

        with torch.no_grad():
            x, y, mic_center, mic_position, speaker_position = batch

            Y_hat, W, Y_bss, W_bss, weight_bss = self(x)
            y_hat = self.stft.inv(Y_hat)
            y_bss = self.stft.inv(Y_bss)

            metrics = self.compute_metrics(y_hat, y)

            val_loss = 0

            if self.loss_type == 'cisdr':
                val_loss = metrics['cisdr_loss']

            if 'kld' in self.loss_type:
                weight = 1. / torch.clamp(Y_hat.real.square() + Y_hat.imag.square(), min=1e-5)
                metrics['kld_loss'] = bss.KLDLoss(
                    Y_hat, 
                    Y_bss, 
                    weight, 
                    weight_bss, 
                    clamp_thres=50,
                    spherical=self.spherical,
                )
                val_loss += metrics['kld_loss'] * self.kld_ratio

            if 'iva' in self.loss_type:
                iva_cisdr_loss = fast_bss_eval.sdr_pit_loss(y_hat, y_bss, clamp_db=25).mean()
                metrics['iva_cisdr_loss'] = iva_cisdr_loss
                val_loss += iva_cisdr_loss

            if 'doa' in self.loss_type:

                idx_1src, idx_2src, doa_1src, doa_2src = est_number_of_sources_via_doa(
                    self.stft(x),
                    #X,
                    mic_position,
                    self.n_fft,
                    n_grid = self.n_grid,
                    n_iter=30,
                    z_only_positive=True,
                )
                doaloss = doa_loss2(
                    doa_2src,
                    W[idx_2src],
                    mic_position[idx_2src],
                    self.n_fft,
                )
                metrics['doa_loss'] = doaloss
                val_loss += doaloss * self.doa_ratio

            metrics['val_loss'] = val_loss

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
