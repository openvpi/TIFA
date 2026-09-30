import pathlib

import click

from lib import logging
from lib.cli import DefaultGroup, csv_list
from lib.config.schema import ConfigurationScope


def _validate_exts(ctx, param, value) -> set[str]:
    try:
        exts = {"." + ext.strip().lower() for ext in value.split(",")}
        if not exts:
            raise ValueError("At least one extension must be provided.")
        return exts
    except Exception as e:
        raise click.BadParameter(f"Invalid extensions: {e}")


def _parse_filemap(path: pathlib.Path, exts: set[str]) -> dict[str, pathlib.Path]:
    """Convert a file or directory path into a ``{identifier: audio_path}`` dict.

    For a single file the key is the stem.  For a directory the keys are
    relative-to-root paths (without extension), preserving any subdirectory
    structure.
    """
    if path.is_file():
        return {path.stem: path}
    if path.is_dir():
        files = [
            f for f in sorted(path.rglob("*"))
            if f.is_file() and f.suffix.lower() in exts
        ]
        filemap = {
            f.relative_to(path).with_suffix("").as_posix(): f
            for f in files
        }
        if not filemap:
            raise FileNotFoundError(f"No audio files found in directory: {path}")
        return filemap
    raise ValueError(f"Invalid path: {path}")


def shared_options(func):
    options = [
        click.option(
            "--model", "-m", required=True,
            type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
            help="Path to model checkpoint.",
        ),
        click.option(
            "--input-formats", default="wav,flac,opus,mp3,aac,ogg",
            show_default=True, callback=_validate_exts,
            help="Comma-separated audio file extensions to scan for (directory mode).",
        ),
        click.option(
            "--output-dir", "-o",
            type=click.Path(file_okay=False, writable=True, path_type=pathlib.Path),
            default=None,
            help="Directory to save output files.  Defaults to the input directory.",
        ),
        click.option(
            "--language", "-l", default=None,
            help="Default language.  Its G2P converters activate; its prefix "
                 "is omitted from output labels.",
        ),
        click.option(
            "--extended-language", "-L",
            default=None, callback=csv_list(str),
            help="Comma-separated additional G2P language tags, in priority order.",
        ),
        click.option(
            "--g2p",
            type=click.Path(exists=True, dir_okay=False, path_type=pathlib.Path),
            default=None,
            help="Custom G2P pipeline config YAML (overrides inference.g2p from config).",
        ),
        click.option(
            "--oov-handling", default="discard",
            type=click.Choice(["raise", "discard", "force"]), show_default=True,
            help="How to handle OOV phonemes: raise (error), discard (drop sample), force (drop OOV paths).",
        ),
        click.option(
            "--skip-handling", default="omit",
            type=click.Choice(["discard", "omit", "preserve"]), show_default=True,
            help="How to handle skipped states (zero-width spans): "
                 "discard (drop sample), omit (exclude intervals), preserve (assign 1ms).",
        ),
        click.option(
            "--skip-penalty", type=float, default=0.5, show_default=True,
            help="Raw cosine-score cost per skipped phoneme, including at sequence boundaries.",
        ),
        click.option(
            "--score-unit", default="levenshtein",
            type=click.Choice(["levenshtein", "word", "none"]), show_default=True,
            help="Pronunciation unit used for MLM scoring.",
        ),
        click.option(
            "--stat", is_flag=True,
            help="Save statistic plots and diagnosis JSON to <output-dir>/statistics/.",
        ),
        click.option(
            "--plot", is_flag=True,
            help="Save per-sample similarity plots next to TextGrid output.",
        ),
        click.option(
            "--batch-size", type=int, default=8, show_default=True,
            help="Batch size for inference.",
        ),
        click.option(
            "--num-workers", type=int, default=2, show_default=True,
            help="Number of dataloader worker processes.",
        ),
        click.option(
            "--precision", default="32-true", show_default=True,
            help="Precision for inference.",
        ),
    ]
    for option in options[::-1]:
        func = option(func)
    return func


def _run_inference(
        scope: int,
        path: pathlib.Path,
        model: pathlib.Path,
        input_formats: set[str],
        output_dir: pathlib.Path | None,
        language: str | None,
        extended_language: list[str] | None,
        g2p: pathlib.Path | None,
        batch_size: int,
        num_workers: int,
        precision: str,
        oov_handling: str,
        skip_handling: str,
        skip_penalty: float = 0.5,
        score_unit: str = "levenshtein",
        stat: bool = False,
        plot: bool = False,
):
    from lightning_utilities.core.rank_zero import rank_zero_info

    from inference.api import load_g2p_config, load_inference_model, run_inference
    from inference.data import AudioTextDataset
    from inference.callbacks import StatisticsCallback, SavePlotCallback, SaveTextGridCallback

    g2p_languages = [language] if language else []
    if extended_language:
        g2p_languages.extend(extended_language)
    g2p_languages = list(dict.fromkeys(g2p_languages))

    filemap = _parse_filemap(path, input_formats)
    if output_dir is None:
        output_dir = path if path.is_dir() else path.parent
    output_dir.mkdir(parents=True, exist_ok=True)

    backend, vocabulary, inference_config = load_inference_model(
        model,
        scope=scope,
    )

    g2p_config = inference_config.g2p
    if g2p is not None:
        g2p_config = load_g2p_config(g2p)
        g2p_root = g2p.parent if g2p.resolve().is_relative_to(model.parent.resolve()) else ""
    elif g2p_config is not None:
        g2p_root = model.parent
    else:
        raise click.UsageError(
            "The model carries no g2p config. Provide one with --g2p."
        )

    dataset = AudioTextDataset(
        filemap=filemap,
        g2p_config=g2p_config,
        g2p_root=g2p_root,
        vocabulary=vocabulary,
        audio_sample_rate=backend.sample_rate,
        language=g2p_languages if g2p_languages else None,
        oov_handling=oov_handling,
    )

    callbacks = [
        SaveTextGridCallback(
            output_dir=output_dir,
            language=language,
            timestep=backend.timestep,
            skip_handling=skip_handling,
        ),
    ]

    if stat:
        callbacks.append(StatisticsCallback(save_dir=output_dir / "statistics"))

    if plot:
        callbacks.append(SavePlotCallback(output_dir=output_dir))

    run_inference(
        backend=backend,
        dataset=dataset,
        callbacks=callbacks,
        batch_size=batch_size,
        num_workers=num_workers,
        precision=precision,
        mode="predict",
        score_unit=score_unit,
        skip_penalty=skip_penalty,
    )
    logging.success("Inference completed.", callback=rank_zero_info)


@click.group(cls=DefaultGroup, help="Run forced alignment inference.")
def main():
    pass


@main.default_command()
@click.argument(
    "path",
    type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=pathlib.Path),
)
@shared_options
def supervised(**kwargs):
    """Align audio with same-basename transcripts: .pfml, then .txt, then .lab."""
    _run_inference(ConfigurationScope.FA, **kwargs)


# @main.command(name="ssl")
# @click.argument(
#     "path",
#     type=click.Path(exists=True, dir_okay=True, file_okay=True, path_type=pathlib.Path),
# )
# @shared_options
def ssl(**kwargs):
    """Self-supervised forced alignment inference."""
    _run_inference(ConfigurationScope.FA_SSL, **kwargs)


if __name__ == "__main__":
    main()
