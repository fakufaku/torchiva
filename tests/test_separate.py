import pytest
import torchiva
from torchiva.separate import get_parser, main

INPUT = "./examples/samples/mix_reverb"
OUTPUT = "./test_sep"
DEVICE = "cpu"


def test_default():
    parser = get_parser()
    args = parser.parse_args([INPUT, OUTPUT + "/default"])
    main(args)


def test_file():
    parser = get_parser()
    args = parser.parse_args(
        [INPUT + "/103-1240-0003_1235-135887-0017.wav", OUTPUT + "/outputfile.wav"]
    )
    main(args)


@pytest.mark.parametrize(
    "algo, model, mic, src",
    [
        # determined
        ("tiss", "nn", 2, 2),
        ("tiss", "laplace", 2, 2),
        ("ip2", "laplace", 2, 2),
        ("tiss", "gauss", 2, 2),
        ("ip2", "gauss", 2, 2),
        # overdetermined
        ("tiss", "nn", 3, 2),
        ("tiss", "laplace", 3, 2),
        ("tiss", "gauss", 3, 2),
        # five
        ("five", "laplace", 2, 1),
        ("five", "gauss", 2, 1),
        ("five", "laplace", 3, 1),
        ("five", "gauss", 3, 1),
    ],
)
def test_separate_command(algo, model, mic, src):
    parser = get_parser()
    output = OUTPUT + f"/{algo}-{model}-{mic}"
    args = parser.parse_args(
        [
            INPUT,
            output,
            "--algo",
            algo,
            "--model-type",
            model,
            "--mic",
            str(mic),
            "--src",
            str(src),
            "--device",
            DEVICE,
        ]
    )
    main(args)
    pass
