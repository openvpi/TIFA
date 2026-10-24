import json
import pathlib

import click

from lib import logging
from lib.cli import DefaultGroup, csv_list, csv_set
from lib.config.io import load_raw_config
from lib.config.schema import ConfigurationScope


def shared_input_options(func):
    options = [
        click.option(
            "--dataset", "-d", required=True,
            type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=pathlib.Path),
            help="Path to the binarized dataset directory.",
        ),
        click.option(
            "--model", "-m", required=True,
            type=click.Path(exists=True, dir_okay=False, file_okay=True, path_type=pathlib.Path),
            help="Path to model checkpoint.",
        ),
        click.option(
            "--prefix", default="valid", show_default=True,
            help="Dataset prefix (e.g. valid, train).",
        ),
    ]
    for option in options[::-1]:
        func = option(func)
    return func


def shared_output_options(func):
    options = [
        click.option(
            "--output-dir", "-o", required=True,
            type=click.Path(file_okay=False, writable=True, path_type=pathlib.Path),
            help="Directory to save evaluation results.",
        ),
        click.option(
            "--plot", is_flag=True,
            help="Save per-sample plots.",
        ),
    ]
    for option in options[::-1]:
        func = option(func)
    return func


def shared_trainer_options(func):
    options = [
        click.option(
            "--skip-penalty", type=float, default=0.5, show_default=True,
            help="Raw cosine-score cost per skipped phoneme during online alignment.",
        ),
        click.option(
            "--batch-size", type=int, default=4, show_default=True,
            help="Batch size for evaluation.",
        ),
        click.option(
            "--concat-size", type=int, default=None,
            help="Concatenate up to N samples per item to test context length.",
        ),
        click.option(
            "--concat-deterministic", is_flag=True, default=False,
            help="Use deterministic grouping for sample concatenation.",
        ),
        click.option(
            "--num-workers", type=int, default=0, show_default=True,
            help="Number of dataloader worker processes.",
        ),
        click.option(
            "--precision", default="32-true", show_default=True,
            help="Precision for evaluation.",
        ),
    ]
    for option in options[::-1]:
        func = option(func)
    return func


def shared_metric_options(func=None, *, unit="ms"):
    if unit == "ms":
        ber_opt = click.option(
            "--ber-tols-ms", "ber_tols", default="50", show_default=True,
            callback=csv_list(int),
            help="Comma-separated BER tolerances in milliseconds.",
        )
    elif unit == "frame":
        ber_opt = click.option(
            "--ber-tols", "ber_tols", default="5", show_default=True,
            callback=csv_list(int),
            help="Comma-separated BER tolerances in frames.",
        )
    else:
        raise ValueError(f"Unknown metric unit: {unit}")

    options = [
        ber_opt,
        click.option(
            "--token-topk", default="20", show_default=True,
            callback=csv_list(int),
            help="Comma-separated worst-k values for per-token metrics.",
        ),
        click.option(
            "--pair-topk", default="20", show_default=True,
            callback=csv_list(int),
            help="Comma-separated worst-k values for pair conjunction metrics.",
        ),
        click.option(
            "--determinacy-power", default=2.0, show_default=True,
            type=float,
            help="Activation power for Determinacy metric.",
        ),
        click.option(
            "--determinacy-width", default=5, show_default=True,
            type=int,
            help="Neighborhood half-width in tokens for Determinacy metric."
                 "  set to negative for unlimited (all tokens).",
        ),
        click.option(
            "--monotonicity-power", default=2.0, show_default=True,
            type=float,
            help="Activation power for Monotonicity metric.",
        ),
        click.option(
            "--monotonicity-width", default=-1, show_default=True,
            type=int,
            help="Neighborhood half-width in tokens for Monotonicity metric."
                 "  set to negative for unlimited (global monotonicity).",
        ),
    ]

    def decorator(f):
        for option in reversed(options):
            f = option(f)
        return f

    if func is None:
        return decorator
    return decorator(func)


def _check_file_and_config(file: pathlib.Path, expected):
    actual = load_raw_config(file, inherit=False, overrides=None)
    if actual != expected:
        raise RuntimeError(
            f"Contents of '{file}' do not match the model configuration. "
            f"The dataset was binarized with different parameters. "
            f"Please re-binarize."
        )


def _check_vocabulary(checkpoint_dir: pathlib.Path, dataset_dir: pathlib.Path):
    ckpt_vocab_path = checkpoint_dir / "vocabulary.json"
    dataset_vocab_path = dataset_dir / "vocabulary.json"
    if not ckpt_vocab_path.is_file():
        raise FileNotFoundError(f"Vocabulary not found at {ckpt_vocab_path}")
    if not dataset_vocab_path.is_file():
        raise FileNotFoundError(f"Vocabulary not found at {dataset_vocab_path}")
    with open(ckpt_vocab_path, "r", encoding="utf8") as f:
        ckpt_vocab = json.load(f)
    with open(dataset_vocab_path, "r", encoding="utf8") as f:
        dataset_vocab = json.load(f)
    if ckpt_vocab != dataset_vocab:
        raise RuntimeError(
            "Vocabulary mismatch between checkpoint and dataset. "
            "The dataset was binarized with a different vocabulary "
            "than the model was trained with."
        )


def _run_online_evaluation(
        scope: int,
        dataset: pathlib.Path,
        model: pathlib.Path,
        prefix: str,
        output_dir: pathlib.Path,
        batch_size: int,
        concat_size: int,
        concat_deterministic: bool,
        num_workers: int,
        precision: str,
        ber_tols: list[int],
        token_topk: list[int],
        pair_topk: list[int],
        plot: bool,
        skip_penalty: float = 0.5,
        determinacy_power: float = 2.0,
        determinacy_width: int = 5,
        monotonicity_power: float = 2.0,
        monotonicity_width: int = -1,
):
    if determinacy_width < 0:
        determinacy_width = None
    if monotonicity_width < 0:
        monotonicity_width = None

    from lightning_utilities.core.rank_zero import rank_zero_info

    from inference.api import (
        load_inference_model,
        run_inference,
    )
    from inference.callbacks import StatisticsCallback, EvaluationMetricsCallback, SavePlotCallback
    from training.data import PhonemeTimingDataset

    backend, vocabulary, inference_config = load_inference_model(model, scope=scope)

    _check_file_and_config(
        dataset / "feature.yaml", inference_config.features.model_dump()
    )
    _check_vocabulary(model.parent, dataset)

    ds = PhonemeTimingDataset(
        data_dir=dataset,
        prefix=prefix,
        max_concat_size=concat_size,
        concat_deterministic=concat_deterministic,
        concat_dynamic_size=False,
        augmentation_config=None,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "summary.json"

    item_paths = [ds.get_metadata("item_paths", i) for i in range(len(ds))]
    callbacks = [
        EvaluationMetricsCallback(
            unit="frame",
            vocab=vocabulary,
            save_path=save_path,
            ber_tols=ber_tols,
            token_topk=token_topk,
            pair_topk=pair_topk,
            determinacy_power=determinacy_power,
            determinacy_width=determinacy_width,
            monotonicity_power=monotonicity_power,
            monotonicity_width=monotonicity_width,
        ),
        StatisticsCallback(
            save_dir=output_dir / "statistics",
            determinacy_power=determinacy_power,
            determinacy_width=determinacy_width,
            monotonicity_power=monotonicity_power,
            monotonicity_width=monotonicity_width,
            identifiers=item_paths,
        ),
    ]
    if plot:
        callbacks.append(
            SavePlotCallback(
                output_dir=output_dir / "plots",
                vocab=vocabulary,
                identifiers=item_paths,
            )
        )

    run_inference(
        backend=backend,
        dataset=ds,
        callbacks=callbacks,
        batch_size=batch_size,
        num_workers=num_workers,
        precision=precision,
        skip_penalty=skip_penalty,
        mode="evaluate",
    )
    logging.success("Online evaluation completed.", callback=rank_zero_info)


def _run_offline_evaluation(
        pred_dir: pathlib.Path,
        gt_dir: pathlib.Path,
        tier_name: str,
        stop_symbols: set[str],
        mismatch_handling: str,
        output_dir: pathlib.Path,
        ber_tols: list[int],
        token_topk: list[int],
        pair_topk: list[int],
        **kwargs,
):
    from lightning_utilities.core.rank_zero import rank_zero_info

    from inference.api import evaluate_offline
    from inference.callbacks import EvaluationMetricsCallback, StatisticsCallback
    from inference.data import PairedDataset, TextGridDataset

    pred_ds = TextGridDataset(pred_dir, tier_name=tier_name, stop_symbols=stop_symbols)
    gt_ds = TextGridDataset(gt_dir, tier_name=tier_name, stop_symbols=stop_symbols)

    paired = PairedDataset(
        pred_dataset=pred_ds,
        gt_dataset=gt_ds,
        mismatch_handling=mismatch_handling,
    )

    output_dir.mkdir(parents=True, exist_ok=True)
    save_path = output_dir / "summary.json"

    statistic_callback = StatisticsCallback(
        save_dir=output_dir / "statistics",
        unit_factor=1000.0,
    )
    metric_callback = EvaluationMetricsCallback(
        unit="ms",
        vocab=paired.vocab,
        save_path=save_path,
        ber_tols=ber_tols,
        token_topk=token_topk,
        pair_topk=pair_topk,
    )

    evaluate_offline(dataset=paired, callbacks=[statistic_callback, metric_callback])
    logging.success("Offline evaluation completed.", callback=rank_zero_info)


@click.group(cls=DefaultGroup, help="Evaluate a model or compare TextGrids.")
def main():
    pass


@main.default_command()
@shared_input_options
@shared_output_options
@shared_trainer_options
@shared_metric_options(unit="frame")
def supervised(**kwargs):
    """Supervised online evaluation on a binarized dataset."""
    _run_online_evaluation(scope=ConfigurationScope.FA, **kwargs)


@main.command(name="ved", help="Evaluate Voice Event Detector frame/event P/R/F1.")
@shared_input_options
@shared_output_options
@click.option("--batch-size", type=click.IntRange(min=1), default=4, show_default=True)
@click.option("--num-workers", type=click.IntRange(min=0), default=0, show_default=True)
@click.option("--precision", default="32-true", show_default=True)
@click.option("--event-collar-ms", type=click.FloatRange(min=0), default=200.0, show_default=True)
@click.option("--require-offset", is_flag=True, help="Match both onset and offset within the event collar.")
def ved(dataset, model, prefix, output_dir, plot, batch_size, num_workers, precision, event_collar_ms, require_offset):
    from lib.config.schema import VoiceEventValidationConfig
    from inference.api import load_voice_event_detector, run_voice_event_evaluation
    from training.data import VoiceEventDataset

    detector, events, config = load_voice_event_detector(model)
    _check_file_and_config(dataset / "feature.yaml", config.features.model_dump())
    _check_vocabulary(model.parent, dataset)
    data = VoiceEventDataset(dataset, prefix)
    run_voice_event_evaluation(
        detector, events, config.features, data, output_dir,
        [str(data.get_metadata("item_paths", index)) for index in range(len(data))],
        batch_size=batch_size, num_workers=num_workers, precision=precision,
        metric_config=VoiceEventValidationConfig(event_collar_ms=event_collar_ms, require_offset=require_offset),
        plot=plot,
    )
    logging.success("Voice event evaluation completed.")


# @main.command(name="ssl")
# @shared_input_options
# @shared_output_options
# @shared_trainer_options
# @shared_metric_options(unit="frame")
def ssl(**kwargs):
    """Self-supervised online evaluation."""
    _run_online_evaluation(scope=ConfigurationScope.FA_SSL, **kwargs)


@main.command(name="offline")
@click.option(
    "--pred", "pred_dir", required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=pathlib.Path),
    help="Directory of predicted TextGrid files.",
)
@click.option(
    "--gt", "gt_dir", required=True,
    type=click.Path(exists=True, dir_okay=True, file_okay=False, path_type=pathlib.Path),
    help="Directory of ground truth TextGrid files.",
)
@click.option(
    "--tier-name", default="phones", show_default=True,
    help="TextGrid tier name to extract.",
)
@click.option(
    "--stop-symbols", default="AP,SP,EP,GS,sil,br,pau", show_default=True,
    callback=csv_set(str),
    help="Comma-separated stop symbols to filter from TextGrid intervals.",
)
@click.option(
    "--mismatch-handling", default="raise",
    type=click.Choice(["raise", "skip"]), show_default=True,
    help="How to handle label mismatches: raise or skip.",
)
@shared_output_options
@shared_metric_options(unit="ms")
def offline(**kwargs):
    """Offline evaluation comparing two sets of TextGrid files."""
    _run_offline_evaluation(**kwargs)


if __name__ == "__main__":
    main()
