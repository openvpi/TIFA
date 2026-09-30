import csv
import pathlib
from dataclasses import dataclass

import librosa
import numpy
from g2pflow import G2PWord

from lib import logging
from lib.audio import load_audio
from lib.feature.pitch import get_pitch_parselmouth
from lib.g2p import build_pipeline_from_config
from lib.g2p_encoding import encode_paths, resolve_phoneme
from lib.vocabulary import Vocabulary, is_stop_symbol, qualify_symbol

from .binarizer_base import (
    BaseBinarizer,
    DataSample,
    MetadataItem,
    find_waveform_file,
)

TEXTS_ITEM_ATTRIBUTES = [
    "paths",  # [P,C] complete aligned candidate columns
    "words",  # [P] semantic word IDs; zero for padding
    "groups",  # [P,C] local pronunciation groups
    "candidates",  # [W,C] validity, including empty candidates
    "f0",  # [T] float32, pitch in Hz
]


@dataclass
class TextMetadataItem(MetadataItem):
    text: str
    g2p_words: list[G2PWord] | None = None
    phones: list[str] | None = None

    @property
    def languages(self) -> list[str]:
        return [tag.strip() for tag in self.language.split("+")]

    @property
    def default_language(self) -> str:
        return self.language.split("+", 1)[0].strip()


class TextOnlyBinarizer(BaseBinarizer[TextMetadataItem]):
    __data_attrs__ = TEXTS_ITEM_ATTRIBUTES

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if self.config.vocabulary.prebuilt_vocab_file is not None:
            self.vocabulary = Vocabulary.from_file(self.config.vocabulary.prebuilt_vocab_file)
        if self.config.g2p is None:
            self.g2p = None
        else:
            self.g2p = build_pipeline_from_config(self.config.g2p)

    def resolve_data_dir(self) -> pathlib.Path:
        return self.config.text_only_data_dir_resolved

    def _is_stop_symbol(self, symbol: str, language: str | None) -> bool:
        vocabulary_config = self.config.vocabulary
        return is_stop_symbol(
            symbol,
            language,
            vocabulary_config.global_symbols,
            vocabulary_config.stop_symbols,
        )

    def _encode_symbol(self, symbol: str, language: str | None) -> int | None:
        if self.vocabulary is None:
            raise RuntimeError("Vocabulary has not been built.")
        token_id = self.vocabulary.encode(symbol, language)
        if token_id is not None:
            return token_id
        if self._is_stop_symbol(symbol, language):
            return None
        raise ValueError(f"Token '{symbol}' is not in the main vocabulary for language " f"'{language}'.")

    def _collect_g2p_symbols(self, words: list[G2PWord], languages: list[str]) -> list[str]:
        vocabulary_config = self.config.vocabulary
        return [
            resolve_phoneme(
                phone, word.language, self.vocabulary,
                languages=languages,
                default_language=languages[0],
                global_symbols=vocabulary_config.global_symbols,
                stop_symbols=vocabulary_config.stop_symbols,
            )[1]
            for word in words
            for reading in word.readings
            for path in reading.paths
            for group in path
            for phone in group.phonemes
        ]

    def filter_metadata_by_vocabulary(
        self,
        metadata_list: list[TextMetadataItem],
    ) -> list[TextMetadataItem]:
        if self.vocabulary is None:
            raise RuntimeError("Vocabulary has not been built.")

        retained: list[TextMetadataItem] = []
        dropped_count = 0
        for item in metadata_list:
            oov_symbols = sorted(
                {
                    symbol
                    for symbol in item.raw_symbols
                    if not self._is_stop_symbol(symbol, item.default_language)
                    and self.vocabulary.encode(symbol, item.default_language) is None
                }
            )
            if not oov_symbols:
                retained.append(item)
                continue

            if item.g2p_words is not None:
                data, _, _ = encode_paths(
                    item.g2p_words, self.vocabulary, "force",
                    default_language=item.default_language,
                    languages=item.languages,
                    global_symbols=self.config.vocabulary.global_symbols,
                    stop_symbols=self.config.vocabulary.stop_symbols,
                )
                if data["candidates"].shape[0] and data["candidates"].any(axis=-1).all():
                    retained.append(item)
                    logging.warning(f"Dropping OOV pronunciation paths from aux item '{item.name}'.")
                    continue

            dropped_count += 1
            # Known-phone samples have no alternatives. A G2P sample must
            # retain at least one complete candidate for every source word.
            logging.warning(
                f"Dropping aux item '{item.name}' from "
                f"'{item.waveform_fn.as_posix()}' because the main vocabulary "
                f"cannot encode: {', '.join(oov_symbols)}."
            )

        if dropped_count:
            logging.warning(
                f"Dropped {dropped_count}/{len(metadata_list)} aux item(s) "
                f"containing tokens absent from the main vocabulary."
            )
        return retained

    def load_metadata(self, subset_dir: pathlib.Path) -> list[TextMetadataItem]:
        index_path = subset_dir / "index.csv"
        with open(index_path, "r", encoding="utf8") as f:
            rows = list(csv.DictReader(f))
        items: list[TextMetadataItem] = []
        for row in rows:
            name = row["name"]
            waveform_fn = find_waveform_file(subset_dir, name)
            if waveform_fn is None:
                continue
            language = row["language"]
            languages = [tag.strip() for tag in language.split("+")]
            if any(not tag for tag in languages):
                raise ValueError(f"Invalid language tags for item '{name}': {language!r}")
            default_language = languages[0]
            estimated_duration = (
                self.get_frame_count(waveform_fn) * self.timestep
            )
            if row.get("phones"):
                raw_phones = row["phones"].split()
                items.append(TextMetadataItem(
                    name=name,
                    language=language,
                    waveform_fn=waveform_fn,
                    estimated_duration=estimated_duration,
                    raw_symbols=[
                        qualify_symbol(phone, default_language, self.config.vocabulary.global_symbols)
                        for phone in raw_phones
                    ],
                    text="",
                    g2p_words=None,
                    phones=raw_phones,
                ))
                continue
            text = row["text"]
            if self.g2p is None:
                raise RuntimeError(
                    f"G2P is not configured but item '{name}' has no 'phones' column. "
                    f"Either configure G2P or add a 'phones' column to the index.csv."
                )
            try:
                g2p_words = self.g2p.convert(text, languages=languages)
            except Exception as e:
                logging.warning(
                    f"G2P failed for item '{name}': {e}"
                )
                continue
            symbols = self._collect_g2p_symbols(g2p_words, languages)
            items.append(TextMetadataItem(
                name=name,
                language=language,
                waveform_fn=waveform_fn,
                estimated_duration=estimated_duration,
                raw_symbols=symbols,
                text=text,
                g2p_words=g2p_words,
            ))
        return items

    def process_item(self, item: TextMetadataItem) -> DataSample:
        if self.vocabulary is None:
            raise RuntimeError("Vocabulary has not been built.")

        length = self.get_frame_count(item.waveform_fn)

        f0 = None
        f0_cfg = getattr(self.config.features, "f0", None)
        if f0_cfg is not None:
            waveform, sr = load_audio(item.waveform_fn)
            if sr != self.config.features.audio_sample_rate:
                waveform = librosa.resample(
                    waveform,
                    orig_sr=sr,
                    target_sr=self.config.features.audio_sample_rate,
                )
            f0, _ = get_pitch_parselmouth(
                waveform,
                self.config.features.audio_sample_rate,
                length,
                hop_size=self.config.features.hop_size,
                f0_min=f0_cfg.f0_min,
                f0_max=f0_cfg.f0_max,
                interp_uv=True,
            )

        if item.phones is not None:
            tokens = [
                tid for ph in item.phones
                if (tid := self._encode_symbol(ph, item.default_language)) is not None
            ]
            size = len(tokens)
            paths = numpy.zeros((max(1, size), 1), dtype=numpy.int64)
            paths[:size, 0] = tokens
            groups = numpy.zeros_like(paths)
            groups[:size, 0] = numpy.arange(1, size + 1)
            data = {
                "paths": paths,
                "words": (paths[:, 0] != 0).astype(numpy.int64),
                "groups": groups,
                "candidates": numpy.ones((1, 1), dtype=numpy.bool_),
            }
        else:
            if item.g2p_words is None:
                raise RuntimeError(f"G2P not run for item '{item.name}'")
            data, lexicon, _ = encode_paths(
                item.g2p_words,
                self.vocabulary,
                "force",
                default_language=item.default_language,
                languages=item.languages,
                global_symbols=self.config.vocabulary.global_symbols,
                stop_symbols=self.config.vocabulary.stop_symbols,
            )
            size = sum(max((len(path["phonemes"]) for path in word), default=0) for word in lexicon)

        if f0 is not None:
            data["f0"] = f0
        return DataSample(
            path=item.waveform_fn.relative_to(self.data_dir).as_posix(),
            name=item.name,
            length=length,
            data=data,
            derived={"paths": size},
        )
