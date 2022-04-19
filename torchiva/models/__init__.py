from .classic import LaplaceModel, GaussModel, NMFModel
from .fnet import FNetModel
from .base import SourceModelBase
from .glu import GLULayer, GLUMask, GLUMask2, MelGLUMask
from .blstm import MultiBLSTMMask, BLSTMMask

source_models = {
    "laplace": LaplaceModel(),
    "gauss": GaussModel(),
    "nmf": NMFModel(),
}
