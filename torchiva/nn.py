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

import enum
from typing import Optional

import torch
from torch import nn

from .five import FIVE
from .auxiva_ip2 import AuxIVA_IP2
from .overiva_iss import OverISS_T
from .models import LaplaceModel
from .beamformer import MVDRBeamformer, MWFBeamformer, GEVBeamformer
from .stft import STFT
from .utils import select_most_energetic


class SepAlgo(enum.Enum):
    AUXIVA_IP2 = "auxiva-ip2"
    OVERISS_T = "overiss_t"
    FIVE = "five"
    MVDR = "mvdr"
    MWF = "mwf"
    GEV = "gev"


class BSSSeparator(nn.Module):
    def __init__(
        self,
        n_fft: int,
        n_iter: Optional[int] = 20,
        hop_length: Optional[int] = None,
        window: Optional[int] = None,
        n_taps: Optional[int] = 0,
        n_delay: Optional[int] = 0,
        n_src: Optional[int] = None,
        algo: Optional[SepAlgo] = SepAlgo.OVERISS_T,
        source_model: Optional[torch.nn.Module] = None,
        proj_back_mic: Optional[int] = 0,
        use_dmc: Optional[bool] = False,
        n_power_iter: Optional[int] = None,
        eps: Optional[float] = 1e-5,
    ):

        super().__init__()

        if source_model is None:
            self.source_model = LaplaceModel()
        else:
            self.source_model = source_model

        self.n_src = n_src
        self.algo = algo

        # other attributes
        self.n_iter = n_iter

        # the stft engine
        self.stft = STFT(n_fft, hop_length=hop_length, window=window)

        # init separator
        if self.algo == "overiss_t":
            self.separator = OverISS_T(
                n_iter=n_iter,
                n_taps=n_taps,
                n_delay=n_delay,
                n_src=n_src,
                model=self.source_model,
                proj_back_mic=proj_back_mic,
                use_dmc=use_dmc,
                eps=eps,
            )

        elif self.algo == "ip2":
            self.separator = AuxIVA_IP2(
                n_iter=n_iter,
                model=self.source_model,
                proj_back_mic=proj_back_mic,
                eps=eps,
            )
            
        elif self.algo == "five":
            self.separator = FIVE(
                n_iter=n_iter,
                model=self.source_model,
                proj_back_mic=proj_back_mic,
                eps=eps,
                n_power_iter=n_power_iter,
            )

        elif self.algo == "mvdr":
            self.separator = MVDRBeamformer(
                self.source_model,
                ref_mic=proj_back_mic,
                eps=eps,
                mvdr_type="stv",
                n_power_iter=n_power_iter,
            )

        elif self.algo == 'mwf':
            self.separator = MWFBeamformer(
                self.source_model,
                ref_mic=proj_back_mic,
                eps=eps,
                time_invariant=True,
            )

        elif self.algo == 'gev':
            self.separator = GEVBeamformer(
                self.source_model,
                ref_mic=proj_back_mic,
                eps=eps,
            )

        else:
            raise NotImplementedError("Selected algorithm is not implemented.")

        

    @property
    def algo(self):
        return self._algo.value

    @algo.setter
    def algo(self, val):
        self._algo = SepAlgo(val)

        if val == SepAlgo.FIVE and self.n_src is not None and self.n_src > 1:
            import warnings

            warnings.warn(
                "Algorithm FIVE can only extract one source. Parameter n_src ignored."
            )


    def forward(self, x, reset=True):

        if hasattr(self.source_model, "reset") and reset:
            # this is required for models with internal state, such a ILRMA
            self.source_model.reset()

        n_chan, n_samples = x.shape[-2:]

        # STFT
        X = self.stft(x)  # copy for back projection (numpy/torch compatible)

        # Separation
        Y = self.separator(X)

        # iSTFT
        y = self.stft.inv(Y)  # (n_samples, n_channels)

        # in case we separated too many sources, select those that have most energy
        if self.n_src is not None and y.shape[-2] > self.n_src:
            y = select_most_energetic(y, num=self.n_src, dim=-2, dim_reduc=-1)

        # zero-padding if necessary 
        if y.shape[-1] < x.shape[-1]:
            y = torch.cat(
                (y, y.new_zeros(y.shape[:-1] + (x.shape[-1] - y.shape[-1],))), dim=-1
            )
        elif y.shape[-1] > x.shape[-1]:
            y = y[..., : x.shape[-1]]

        return y