import torchiva
import pytest


MODEL_URL = (
    "https://raw.githubusercontent.com/fakufaku/torchiva/master/trained_models/tiss"
)


def test_load_from_path_legacy():
    sep = torchiva.load_separator_model(
        "trained_models/tiss/model_weights.ckpt",
        "trained_models/tiss/model_config.yaml",
    )


# call with URL twice because it is not downloaded the second time
@pytest.mark.parametrize(
    "path",
    ["./trained_models/tiss", MODEL_URL, MODEL_URL],
)
def test_loader(path):
    sep = torchiva.load_separator(path)


if __name__ == "__main__":
    test_load_from_path_legacy()
    test_loader("./trained_models/tiss")
    test_loader(MODEL_URL)
    test_loader(MODEL_URL)
