import pathlib
from collections.abc import Mapping

from lib import logging
from lib.config.formatter import format_model
from lib.config.io import load_raw_config
from lib.config.schema import BinarizerConfig, RootConfig
from lib.vocabulary import Vocabulary, VocabularyBuilder

__all__ = [
    "load_config_for_binarization",
    "build_vocab_from_datasets",
    "build_shared_vocab",
    "binarize_datasets",
]


def load_config_for_binarization(
        config_path: pathlib.Path,
        scope: int = 0,
        overrides: list[str] = None
) -> BinarizerConfig:
    config = load_raw_config(config_path, inherit=True, overrides=overrides)
    config = RootConfig.model_validate(config, scope=scope)
    config.resolve(scope_mask=scope)
    config.check(scope_mask=scope)
    print(format_model(config.binarizer))
    return config.binarizer


def build_shared_vocab(vocab_config, metadata_list) -> tuple[Vocabulary, Mapping, VocabularyBuilder]:
    prebuilt = None
    if vocab_config.prebuilt_vocab_file is not None:
        prebuilt = Vocabulary.from_file(vocab_config.prebuilt_vocab_file)
    builder = VocabularyBuilder(
        global_symbols=vocab_config.global_symbols,
        stop_symbols=vocab_config.stop_symbols,
        merged_groups=vocab_config.merged_groups,
        prebuilt_vocab=prebuilt,
    )
    for item in metadata_list:
        builder.add(item.raw_symbols, default_language=item.language)
    vocab = builder.build()
    counter = builder.counter()
    return vocab, counter, builder


def build_vocab_from_datasets(
        config: BinarizerConfig,
        binarizer_classes: list[type],
):
    binarizers = [cls(config=config) for cls in binarizer_classes]
    main_metadata = binarizers[0].collect_metadata()
    logging.info(
        f"Collected {len(main_metadata)} main metadata items "
        f"from '{binarizers[0].data_dir.as_posix()}' "
        f"({binarizers[0].__class__.__name__})."
    )
    if not main_metadata:
        raise RuntimeError("No metadata items found in the main dataset.")
    if len(binarizers) == 1:
        binarizers[0].build_vocabulary(main_metadata)
        logging.success("Vocabulary built.")
        return
    vocabulary, counter, builder = build_shared_vocab(
        config.vocabulary,
        main_metadata,
    )
    for binarizer in binarizers:
        binarizer.vocabulary = vocabulary
        binarizer.vocab_builder = builder
    binarizers[0].save_vocab_plot(counter)

    for binarizer in binarizers[1:]:
        metadata = binarizer.collect_metadata()
        retained = binarizer.filter_metadata_by_vocabulary(metadata)
        logging.info(
            f"Validated {len(metadata)} aux metadata items from "
            f"'{binarizer.data_dir.as_posix()}': {len(retained)} retained."
        )
    logging.success("Vocabulary built and plot saved.")


def binarize_datasets(
        config: BinarizerConfig,
        binarizer_classes: list[type],
        eval_mode: bool = False
):
    binarizers = [
        binarizer_classes[0](config=config, eval_mode=eval_mode),
        *[
            cls(config=config, eval_mode=eval_mode, aux_mode=True)
            for cls in binarizer_classes[1:]
        ]
    ]

    if len(binarizers) == 1:
        logging.info(f"Starting binarizer: {binarizers[0].__class__.__name__}.")
        binarizers[0].process()
        logging.success("Binarization completed.")
        return

    main_metadata = binarizers[0].collect_metadata()
    logging.info(
        f"Collected {len(main_metadata)} main metadata items "
        f"from '{binarizers[0].data_dir.as_posix()}' "
        f"({binarizers[0].__class__.__name__})."
    )
    if not main_metadata:
        raise RuntimeError("No metadata items found in the main dataset.")

    # The model vocabulary is defined only by supervised main data. Aux-only
    # symbols must never expand the output space.
    shared_vocab, counter, builder = build_shared_vocab(
        config.vocabulary,
        main_metadata,
    )
    for b in binarizers:
        b.vocabulary = shared_vocab
        b.vocab_builder = builder

    # Collect aux metadata with the shared vocabulary, then validate before writing.
    per_metadata = [main_metadata]
    for b in binarizers[1:]:
        metadata = b.collect_metadata()
        retained = b.filter_metadata_by_vocabulary(metadata)
        per_metadata.append(retained)
        logging.info(
            f"Aux dataset '{b.data_dir.as_posix()}': "
            f"retained {len(retained)}/{len(metadata)} items "
            f"after main-vocabulary validation."
        )

    # Main binarizer handles plot.
    binarizers[0].save_vocab_plot(counter)

    # Each binarizer builds dataset independently
    for b, metadata in zip(binarizers, per_metadata):
        b.build_dataset(metadata)

    # Each binarizer saves its own auxiliary files
    for b in binarizers:
        b.save_auxiliary_files()

    logging.success("Binarization completed.")
