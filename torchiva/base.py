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

import math
from enum import Enum
from typing import Optional, Union

import torch

from .models import LaplaceModel
from .parameters import eps_models


class Window(Enum):
    CUSTOM = None
    BARTLETT = "bartlett"
    BLACKMAN = "blackman"
    HAMMING = "hamming"
    HANN = "hann"


window_types = [s for s in Window._value2member_map_ if s is not None]


class SourceModelBase(torch.nn.Module):
    """
    An abstract class to represent source models

    Parameters
    ----------
    X: numpy.ndarray or torch.Tensor, shape (..., n_frequencies, n_frames)
        STFT representation of the signal

    Returns
    -------
    P: numpy.ndarray or torch.Tensor, shape (..., n_frequencies, n_frames)
        The inverse of the source power estimate
    """

    def __init__(self):
        super().__init__()

    def reset(self):
        """
        The reset method is intended for models that have some internal state
        that should be reset for every new signal.

        By default, it does nothing and should be overloaded when needed by
        a subclass.
        """
        pass


class DRBSSBase(torch.nn.Module):
    def __init__(
        self,
        n_iter: Optional[int] = 10,
        n_taps: Optional[int] = 0,
        n_delay: Optional[int] = 0,
        n_src: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
        proj_back_mic: Optional[int] = 0,
        use_dmc: Optional[bool] = False,
        eps: Optional[float] = 1e-5,
    ):
        super().__init__()

        self._n_taps = n_taps
        self._n_delay = n_delay
        self._n_iter = n_iter
        self._n_src = n_src
        self._proj_back_mic = proj_back_mic
        self._use_dmc = use_dmc

        if eps is None:
            self._eps = eps_models["laplace"]
        else:
            self._eps = eps

        if model is None:
            self.model = LaplaceModel()
        else:
            self.model = model
        assert callable(self.model)

        # metrology
        self.checkpoints_list = []

    def _set_params(self, **kwargs):
        for (key, value) in kwargs.items():
            if value is None:
                kwargs[key] = getattr(self, key)

        return kwargs.values()

    def _reset(self, model):
        if hasattr(model, "reset"):
            model.reset()

    @property
    def n_iter(self):
        return self._n_iter

    @property
    def n_taps(self):
        return self._n_taps

    @property
    def n_delay(self):
        return self._n_delay

    @property
    def n_src(self):
        return self._n_src

    @property
    def proj_back_mic(self):
        return self._proj_back_mic

    @property
    def use_dmc(self):
        return self._use_dmc

    @property
    def eps(self):
        return self._eps


class BFBase(torch.nn.Module):
    def __init__(
        self,
        mask_model: torch.nn.Module,
        ref_mic: Optional[int] = 0,
        eps: Optional[float] = 1e-5,
    ):
        super().__init__()

        self.mask_model = mask_model

        self._ref_mic = ref_mic
        self._eps = eps

        # just so that training works
        self.fake = torch.nn.Parameter(torch.zeros(1))

    def _set_params(self, **kwargs):
        for (key, value) in kwargs.items():
            if value is None:
                kwargs[key] = getattr(self, key)

        return kwargs.values()

    @property
    def ref_mic(self):
        return self._ref_mic

    @property
    def eps(self):
        return self._eps
