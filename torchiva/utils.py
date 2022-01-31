from typing import Dict, Optional

from torch import nn


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
