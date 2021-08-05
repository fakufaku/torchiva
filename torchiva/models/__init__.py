from .classic import LaplaceModel, GaussModel, NMFModel
from .fnet import FNetModel
from .base import SourceModelBase
from .simple import SimpleModel

source_models = {
    "laplace": LaplaceModel(),
    "gauss": GaussModel(),
    "nmf": NMFModel(),
}
