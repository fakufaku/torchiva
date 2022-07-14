from pathlib import Path
from typing import Union

import torch
import yaml

from .nn import BSSSeparator


def load_separator_model(
    ckpt_path: Union[Path, str], config_path: Union[Path, str], **kwargs
) -> BSSSeparator:
    """
    Loads pre-trained weights into a ``BSSSeparator`` object.

    Parameters
    ----------
    ckpt_path: str or Path object
        Path to model weight checkpoint
    config_path: str or Path object
        Path to yaml file containing the ``BSSSeparator`` object parameters
    **kwargs:
        Extra parameters to be added or changed in the model config

    Returns
    -------
    A ``BSSSeparator`` object loaded with the pre-trained weights
    """
    with open(config_path, "r") as f:
        model_config = yaml.safe_load(f)

    model_config.update(kwargs)

    state_dict = torch.load(ckpt_path)
    separator = BSSSeparator(**model_config)
    separator.load_state_dict(state_dict)

    return separator
