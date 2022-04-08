import abc
import math
from enum import Enum
from typing import Optional, Union

import torch 

from .models import LaplaceModel
from .parameters import eps_models


class IVABase(torch.nn.Module):
    def __init__(
        self,
        n_iter: int,
        n_taps: Optional[int] = 0,
        n_delay: Optional[int] = 0,
        n_src: Optional[int] = None,
        model: Optional[torch.nn.Module] = None,
        proj_back_mic: Optional[int] = 0,
        use_dmc: Optional[bool] = False,
        eps: Optional[float] = 1e-5,
    ):
        super().__init__()

        self.n_taps = n_taps
        self.n_delay = n_delay
        self.n_iter = n_iter
        self.n_src = n_src
        self.proj_back_mic = proj_back_mic
        self.use_dmc = use_dmc
        
        if eps is None:
            self.eps = eps_models["laplace"]
        else:
            self.eps = eps

        if model is None:
            self.model = LaplaceModel()
        else:
            self.model = model
        assert callable(self.model)

        # metrology
        self.checkpoints_list = []



    def _forward(self, X, **kwargs):
        pass

    def _set_params(self, **kwargs):
        
        for (key, value) in kwargs.items():
            if value is None:
                kwargs[key] = getattr(self, key)

        return kwargs.values()

    def _preprocess(self):
        pass
    
    def _one_iteration(self):
        pass

    def _projection_back(self, A, proj_back_mic):
        pass
    

