from .blstm import BLSTMMask, MaskMVDRSupport
from .classic import GaussModel, LaplaceModel, NMFModel
from .glu import GLUMask
from .simple import SimpleModel

source_models = {
    "laplace": LaplaceModel(),
    "gauss": GaussModel(),
    "nmf": NMFModel(),
}
