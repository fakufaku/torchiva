import os
from pathlib import Path
from typing import Union, Optional

import torch
import yaml

from .nn import BSSSeparator

from urllib.parse import urlparse
from urllib.request import urlretrieve

WEIGHTS_FN = "model_weights.ckpt"
CONFIG_FN = "model_config.yaml"


def get_model_filenames(path):
    ckpt_fn = path / f"{WEIGHTS_FN}"
    yaml_fn = path / f"{CONFIG_FN}"
    return ckpt_fn, yaml_fn


def urljoin(part1, part2):
    return "/".join([part1.strip("/"), part2])


def get_model_from_url(url):
    path = Path(url)
    model_name = path.name
    yaml_url = urljoin(url, CONFIG_FN)
    ckpt_url = urljoin(url, WEIGHTS_FN)
    print(url)
    print(yaml_url)
    print(ckpt_url)

    model_path = Path.home() / f".torchiva_models/{model_name}"
    ckpt_fn, yaml_fn = get_model_filenames(model_path)

    ckpt_fn.parent.mkdir(exist_ok=True, parents=True)

    for fn, url in ((ckpt_fn, ckpt_url), (yaml_fn, yaml_url)):
        if not fn.exists():
            urlretrieve(url, filename=fn)

    return ckpt_fn, yaml_fn


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
    ckpt_path = Path(ckpt_path)
    config_path = Path(config_path)

    if not ckpt_path.exists():
        raise ValueError(f"The model weights file {ckpt_path} does not exist")

    if not config_path.exists():
        raise ValueError(f"The model config file {config_path} does not exist")

    with open(config_path, "r") as f:
        model_config = yaml.safe_load(f)

    model_config.update(kwargs)

    state_dict = torch.load(ckpt_path)
    separator = BSSSeparator(**model_config)
    separator.load_state_dict(state_dict)

    return separator


def load_separator(path: str, **kwargs) -> BSSSeparator:
    """
    Loads pre-trained weights into a ``BSSSeparator`` object.

    Parameters
    ----------
    path: str or url
        Path/URL to the folder containing the model_weights.ckpt and
        model_config.yaml files
    **kwargs:
        Extra parameters to be added or changed in the model config

    Returns
    -------
    A ``BSSSeparator`` object loaded with the pre-trained weights
    """

    result = urlparse(path)
    path_obj = Path(path)
    if result.scheme in ("http", "https"):
        ckpt_path, config_path = get_model_from_url(path)

    elif path_obj.exists():
        if path_obj.is_dir():
            ckpt_path, config_path = get_model_filenames(path_obj)
        else:
            raise ValueError(f"The path {path} is not a folder")

    else:
        raise ValueError("The argument must be an existing local path or URL")

    return load_separator_model(ckpt_path, config_path, **kwargs)
