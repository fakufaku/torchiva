from typing import Dict, Optional

import torch
from torch import nn


def select_most_energetic(
    x: torch.Tensor, num: int, dim: Optional[int] = -2, dim_reduc: Optional[int] = -1
):
    """
    Selects the `num` indices with most power

    Parametes
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


def module_from_config(name: str, kwargs: Optional[Dict] = None, **other):
    """
    Get a model by its name

    Parameters
    ----------
    name: str
        Name of the model class
    kwargs: dict
        A dict containing all the arguments to the model
    """

    if kwargs is None:
        kwargs = {}

    parts = name.split(".")
    obj = None

    if len(parts) == 1:
        raise ValueError("Can't find object without module name")

    else:
        module = __import__(".".join(parts[:-1]), fromlist=(parts[-1],))
        if hasattr(module, parts[-1]):
            obj = getattr(module, parts[-1])

    if obj is not None:
        return obj(**kwargs)
    else:
        raise ValueError(f"The model {name} could not be found")
