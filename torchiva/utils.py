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
from typing import Dict, Optional

import numpy as np
import torch
import yaml
from torch import nn


def select_most_energetic(
    x: torch.Tensor, num: int, dim: Optional[int] = -2, dim_reduc: Optional[int] = -1
):
    """
    Selects the `num` indices with most power

    Parameters
    ----------
    x: torch.Tensor  (n_batch, n_channels, n_samples)
        The input tensor
    num: int
        The number of signals to select
    dim:
        The axis where the selection should occur
    dim_reduc:
        The axis where to perform the reduction
    """

    power = x.abs().square().mean(axis=dim_reduc, keepdim=True)

    index = torch.argsort(power.transpose(dim, -1), axis=-1, descending=True)
    index = index[..., :num]

    # need to broadcast to target size
    x_tgt = x.transpose(dim, -1)[..., :num]
    _, index = torch.broadcast_tensors(x_tgt, index)

    # reorder index axis
    index = index.transpose(dim, -1)

    ret = torch.gather(x, dim=dim, index=index)
    return ret


from typing import Dict, List, Optional, Tuple, Union


def import_name(name: str):

    parts = name.split(".")

    if len(parts) == 1:
        return __import__(parts[0])

    module_name = ".".join(parts[:-1])
    target_name = parts[-1]

    module = __import__(module_name, fromlist=(target_name,))
    if hasattr(module, target_name):
        return getattr(module, target_name)
    else:
        raise ImportError(f"Could not import {target_name} from {module_name}")


def instantiate(
    name: str, args: Optional[Union[Tuple, List]] = None, kwargs: Optional[Dict] = None
):
    """
    Get a model by its name
    Parameters
    ----------
    name: str
        Name of the model class
    kwargs: dict
        A dict containing all the arguments to the model
    """

    if args is None:
        args = tuple()

    if kwargs is None:
        kwargs = {}

    obj = import_name(name)

    return obj(*args, **kwargs)
