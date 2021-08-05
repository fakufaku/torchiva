import abc
import math
from enum import Enum
from typing import Optional, Union

import torch as pt


class Window(Enum):
    CUSTOM = None
    BARTLETT = "bartlett"
    BLACKMAN = "blackman"
    HAMMING = "hamming"
    HANN = "hann"


window_types = [s for s in Window._value2member_map_ if s is not None]


class STFTBase(abc.ABC):
    def __init__(
        self,
        n_fft: int,
        hop_length: Optional[int] = None,
        window: Optional[Union[Window, str]] = None,
    ):
        self._n_fft = n_fft
        self._n_freq = n_fft // 2 + 1
        self._hop_length = hop_length

        if window is None:
            self._window_type = Window.HAMMING
        else:
            self._window_type = Window(window)

        # defer window creation to derived class
        self._window = None

    def _get_n_frames(self, n_samples):
        n_hop = math.floor(n_samples / self.hop_length)
        return n_hop + 1

    @property
    def n_fft(self):
        return self._n_fft

    @property
    def hop_length(self):
        return self._hop_length

    @property
    def n_freq(self):
        return self._n_freq

    @property
    def window(self):
        return self._window

    @property
    def window_type(self):
        return self._window_type

    @property
    def window_name(self):
        return self._window_type.value

    @abc.abstractmethod
    def _make_window(self, dtype):
        pass

    @abc.abstractmethod
    def _forward(self, x):
        pass

    @abc.abstractmethod
    def _backward(self, x):
        pass

    def __call__(self, x):
        self._make_window(x)
        return self._forward(x)

    def inv(self, x):
        self._make_window(x)
        return self._backward(x)
