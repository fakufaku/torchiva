from .classic import LaplaceModel, GaussModel, NMFModel

source_models = {
    "laplace": LaplaceModel(),
    "gauss": GaussModel(),
    "nmf": NMFModel(),
}
