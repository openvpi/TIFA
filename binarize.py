import pathlib

import click

from lib import logging
from lib.cli import DefaultGroup
from lib.config.schema import ConfigurationScope


def shared_options(func):
    options = [
        click.option(
            "--config", type=click.Path(
                exists=True, dir_okay=False, file_okay=True, readable=True, path_type=pathlib.Path
            ),
            required=True,
            help="Path to the configuration file."
        ),
        click.option(
            "--override", multiple=True,
            type=click.STRING, required=False,
            help="Override configuration values in dotlist format."
        ),
        click.option(
            "--vocab", "vocab_only", is_flag=True, default=False, show_default=True,
            help="Build vocabulary from metadata, save a distribution plot when available, and exit."
        ),
        click.option(
            "--eval", "eval_mode", is_flag=True, default=False, show_default=True,
            help="Evaluation mode: the whole dataset will be processed as validation set."
        ),
    ]
    for option in options[::-1]:
        func = option(func)
    return func


def _run_binarization(
        config: pathlib.Path, override: list[str],
        vocab_only: bool, eval_mode: bool,
        scope: int,
):
    from preprocessing.api import (
        load_config_for_binarization, build_vocab_from_datasets, binarize_datasets
    )
    from preprocessing.events_binarizer import VoiceEventBinarizer
    from preprocessing.phonemes_binarizer import PhonemeTimingBinarizer
    from preprocessing.texts_binarizer import TextOnlyBinarizer

    config_obj = load_config_for_binarization(config, scope=scope, overrides=override)

    if scope == ConfigurationScope.FA:
        binarizer_classes = [PhonemeTimingBinarizer]
        if config_obj.text_only_data_dir is not None:
            binarizer_classes.append(TextOnlyBinarizer)
    elif scope == ConfigurationScope.FA_SSL:
        binarizer_classes = [TextOnlyBinarizer]
    elif scope == ConfigurationScope.VED:
        binarizer_classes = [VoiceEventBinarizer]
    else:
        raise ValueError(f"Unknown scope: {scope}")

    if vocab_only:
        build_vocab_from_datasets(
            config=config_obj,
            binarizer_classes=binarizer_classes,
        )
        return

    if eval_mode:
        logging.debug("Using evaluation mode.")

    binarize_datasets(
        config=config_obj,
        binarizer_classes=binarizer_classes,
        eval_mode=eval_mode
    )


@click.group(cls=DefaultGroup)
def main():
    pass


@main.default_command(help="Binarize phoneme-timing datasets for supervised learning.")
@shared_options
def supervised(config, override, vocab_only, eval_mode):
    _run_binarization(
        config=config, override=override,
        vocab_only=vocab_only, eval_mode=eval_mode,
        scope=ConfigurationScope.FA,
    )


@main.command(name="ved", help="Binarize phoneme-timing data for Voice Event Detector.")
@shared_options
def ved(config, override, vocab_only, eval_mode):
    _run_binarization(
        config=config, override=override,
        vocab_only=vocab_only, eval_mode=eval_mode,
        scope=ConfigurationScope.VED,
    )


# @main.command(name="ssl", help="Binarize text-only dataset for self-supervised learning.")
# @shared_options
def ssl(config, override, vocab_only, eval_mode):
    _run_binarization(
        config=config, override=override,
        vocab_only=vocab_only, eval_mode=eval_mode,
        scope=ConfigurationScope.FA_SSL,
    )


if __name__ == "__main__":
    main()
