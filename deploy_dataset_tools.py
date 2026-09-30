"""Export a model and its G2P resources for the dataset-tools Tifa application.

The application in https://github.com/openvpi/dataset-tools reads a model
directory (``config.json``, ``vocabulary.json`` and five ONNX graphs) and a
dictionary directory (the pronunciation dictionaries of this repository and,
optionally, the English out-of-vocabulary model).  :func:`deploy_model` writes
the first one; this script adds the second one and prints where both belong.
"""

import pathlib
import shutil

import click

from deployment.api import deploy_model
from inference.api import load_inference_model
from lib import logging
from lib.config.schema import ConfigurationScope

REPOSITORY_ROOT = pathlib.Path(__file__).resolve().parent

# Dictionaries of the G2P pipeline; see configs/g2p.yaml.
DICTIONARIES = (
    "ds-zh-pinyin-lite.txt",
    "jyutping_dict.txt",
    "japanese_dict_full.txt",
    "ds_cmudict-07b.txt",
)

# Optional English out-of-vocabulary model, see docs/G2P.md.
ENGLISH_OOV = "LstmG2p-Eng"


@click.command(help="Export a dataset-tools-ready directory with model/ and dict/.")
@click.option(
    "-m", "--model", required=True,
    type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
    help="Checkpoint with config.yaml and vocabulary.json beside it.",
)
@click.option(
    "-o", "--save-dir", required=True,
    type=click.Path(file_okay=False, path_type=pathlib.Path),
    help="Directory for model/Tifa (ONNX graphs) and dict (dictionaries).",
)
@click.option(
    "--opset-version", type=click.IntRange(min=18, max=20), default=18, show_default=True,
    help="ONNX opset; 18 supports scoring reductions and DirectML.",
)
def main(model: pathlib.Path, save_dir: pathlib.Path, opset_version: int):
    backend, vocabulary, _ = load_inference_model(model, scope=ConfigurationScope.FA)
    model_dir = save_dir / "model" / "Tifa"
    deploy_model(
        backend, vocabulary, model_dir, opset_version=opset_version,
        config_path=model.parent / "config.yaml",
    )

    dict_dir = save_dir / "dict"
    dict_dir.mkdir(parents=True, exist_ok=True)
    for name in DICTIONARIES:
        source = REPOSITORY_ROOT / "dictionaries" / name
        if not source.is_file():
            raise click.ClickException(f"Missing dictionary: '{source.as_posix()}'.")
        shutil.copyfile(source, dict_dir / name)
    logging.success(f"Copied {len(DICTIONARIES)} dictionaries to '{dict_dir.as_posix()}'.")

    oov_dir = REPOSITORY_ROOT / "assets" / ENGLISH_OOV
    if oov_dir.is_dir():
        shutil.copytree(oov_dir, dict_dir / ENGLISH_OOV, dirs_exist_ok=True)
        logging.success(f"Copied '{ENGLISH_OOV}' to '{dict_dir.as_posix()}'.")
    else:
        logging.warning(
            f"'{oov_dir.as_posix()}' is missing. English words that are not in the "
            f"pronunciation dictionary will be reported as out of vocabulary."
        )

    logging.info(f"Package ready in '{save_dir.as_posix()}'.")
    logging.info(f"Copy '{model_dir.as_posix()}' to '<dataset-tools>/bin/model/Tifa'.")
    logging.info(
        f"Copy the files of '{dict_dir.as_posix()}' into '<dataset-tools>/bin/dict', which "
        f"already contains the 'mandarin' and 'cantonese' folders of cpp-pinyin."
    )


if __name__ == "__main__":
    main()
