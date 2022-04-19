# Copyright (c) 2022 Robin Scheibler
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

import torch as pt


class SourceModelBase(pt.nn.Module):
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
