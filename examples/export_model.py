import argparse
from pathlib import Path
from test import load_ensemble_model, load_model

import torch
import yaml

from separation_model import WSJModel

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run one algorithm over the dataset")
    parser.add_argument("output", type=Path, help="Path to output folder")
    parser.add_argument("folder", type=Path, help="Path to lightning experiment folder")
    parser.add_argument(
        "epochs",
        nargs="+",
        help="Epoch number to be evaluated. Multiple values are ensembled.",
    )
    args = parser.parse_args()
    hparams_path = args.folder / "hparams.yaml"
    hparams, separator = load_ensemble_model(hparams_path, args.epochs)

    model_config = hparams["config"]["model"]
    if "ref_mic" in model_config:
        ref_mic = model_config.pop("ref_mic")
        model_config["proj_back_mic"] = ref_mic
    state_dict = separator.state_dict()

    args.output.mkdir(parents=True, exist_ok=True)

    with open(args.output / "model_config.yaml", "w") as f:
        yaml.dump(model_config, f)
    torch.save(state_dict, args.output / "model_weights.ckpt")
