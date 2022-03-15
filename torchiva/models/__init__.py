from .classic import LaplaceModel, GaussModel, NMFModel
from .fnet import FNetModel
from .base import SourceModelBase
from .simple import SimpleModel, SimpleModel2
from .glu import GLULayer, GLUMask, GLUMask2, MelGLUMask
from .blstm import MultiBLSTMMask, BLSTMMask

source_models = {
    "laplace": LaplaceModel(),
    "gauss": GaussModel(),
    "nmf": NMFModel(),
}
