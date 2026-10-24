import pathlib

import click

from lib.cli import DefaultGroup
from lib.config.schema import ConfigurationScope


def shared_options(func):
    options = [
        click.option(
            "-m", "--model", required=True,
            type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
            help="Checkpoint with config.yaml and vocabulary.json beside it.",
        ),
        click.option(
            "-o", "--save-dir", required=True,
            type=click.Path(file_okay=False, path_type=pathlib.Path),
            help="Directory for ONNX graphs, config.json and vocabulary.json.",
        ),
        click.option(
            "--opset-version", type=click.IntRange(min=18, max=20), default=18, show_default=True,
            help="ONNX opset for DirectML-compatible graphs.",
        ),
    ]
    for option in options[::-1]:
        func = option(func)
    return func


@click.group(cls=DefaultGroup, help="Export models as separate ONNX graphs.")
def main():
    pass


@main.default_command()
@shared_options
def supervised(model: pathlib.Path, save_dir: pathlib.Path, opset_version: int):
    """Export supervised forced alignment as separate ONNX graphs."""
    from deployment.api import deploy_model
    from inference.api import load_inference_model

    backend, vocabulary, _ = load_inference_model(model, scope=ConfigurationScope.FA)
    deploy_model(
        backend, vocabulary, save_dir, opset_version=opset_version,
        config_path=model.parent / "config.yaml",
    )


@main.command()
@shared_options
def ved(model: pathlib.Path, save_dir: pathlib.Path, opset_version: int):
    """Export Voice Event Detection spectrogram and frame classification graphs."""
    from deployment.api import deploy_voice_event_detector
    from inference.api import load_voice_event_detector

    detector, events, inference_config = load_voice_event_detector(model)
    deploy_voice_event_detector(
        detector, events, inference_config.features, save_dir, opset_version=opset_version,
    )


if __name__ == "__main__":
    main()
