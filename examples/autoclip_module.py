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

import bisect
from typing import Tuple

import torch


def grad_norm(module: torch.nn.Module):
    """
    Helper function that computes the gradient norm
    """
    total_norm = 0
    for p in module.parameters():
        if p.grad is not None:
            param_norm = p.grad.data.norm(2)
            total_norm += param_norm.item() ** 2
    total_norm = total_norm ** 0.5
    return total_norm


class AutoClipper:
    """
    This class can be used to automatically adjust the threshold used 
    to clip the gradient
    Parameters
    ----------
    p: int
        The percentile between 0 and 100 where we choose the threshold
        in the gradient history.
    """

    def __init__(self, p: float):
        self.autoclip_p = p / 100
        self.grad_norm_history = []

    def __call__(self, module: torch.nn.Module) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Computes the norm of the gradient of module, adjusts the clipping
        threshold, and apply the clipping to the weights of module.
        """

        # compute the gradient norm and insert in the list
        # in sorted order
        gnorm = grad_norm(module)
        bisect.insort(self.grad_norm_history, gnorm)

        # select the pth percentile
        index = int(self.autoclip_p * len(self.grad_norm_history))
        if index == len(self.grad_norm_history):
            index -= 1
        grad_clip_norm = self.grad_norm_history[index]

        # perform the clipping
        torch.nn.utils.clip_grad_norm_(module.parameters(), max_norm=grad_clip_norm)

        return gnorm, grad_clip_norm