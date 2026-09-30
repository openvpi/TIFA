import pathlib
from typing import Any, Literal

import librosa
import textgrid
import torch
import torch.utils.data
from g2pflow import G2PPipelineConfig

from lib import logging
from lib.audio import load_audio
from lib.g2p import build_pipeline_from_config
from lib.g2p_encoding import G2PEncodingError, encode_paths
from lib.vocabulary import Vocabulary
from training.data import collate_nd


def _skip(identifier: str, reason: str) -> dict[str, Any]:
    return {"skip": True, "identifier": identifier, "warning": f"Skipping '{identifier}': {reason}"}


class AudioTextDataset(torch.utils.data.Dataset):
    """Pairs audio files with text files for forced alignment inference.

    Takes a filemap ``{identifier: audio_path}``.  In ``__getitem__``
    the paired ``.txt`` file is located alongside the audio and G2P
    conversion produces complete word-candidate grids.
    """

    def __init__(
        self,
        filemap: dict[str, pathlib.Path],
        g2p_config: G2PPipelineConfig,
        g2p_root: str | pathlib.Path,
        vocabulary: Vocabulary,
        audio_sample_rate: int,
        language: str | list[str] | None = None,
        oov_handling: Literal["raise", "discard", "force"] = "discard",
    ):
        self.g2p_config = g2p_config
        self.g2p_root = g2p_root
        self._g2p_pipeline = None
        self.vocabulary = vocabulary
        self.sample_rate = audio_sample_rate
        if isinstance(language, str):
            language = [language]
        self.language = language
        self.oov_handling = oov_handling

        self.items: list[tuple[pathlib.Path, str]] = [
            (audio_path, identifier)
            for identifier, audio_path in sorted(filemap.items())
        ]
        if not self.items:
            raise ValueError("Empty filemap")

    def _get_g2p(self):
        if self._g2p_pipeline is None:
            self._g2p_pipeline = build_pipeline_from_config(
                self.g2p_config, root_path=self.g2p_root,
            )
        return self._g2p_pipeline

    def __len__(self):
        return len(self.items)

    def __getitem__(self, idx):
        g2p_pipeline = self._get_g2p()
        audio_path, identifier = self.items[idx]

        text_path = audio_path.with_suffix(".txt")
        if not text_path.is_file():
            text_path = audio_path.with_suffix(".lab")
        if not text_path.is_file():
            return _skip(identifier, "No paired text file")

        with open(text_path, "r", encoding="utf8") as f:
            text = f.read().strip()
        if not text:
            return _skip(identifier, "Empty text")

        try:
            g2p_words = g2p_pipeline.convert(
                text, languages=self.language,
            )
        except Exception as e:
            return _skip(identifier, f"G2P failed: {e}")

        try:
            data, lexicon, texts = encode_paths(
                g2p_words, self.vocabulary, self.oov_handling,
                languages=self.language,
            )
        except G2PEncodingError as e:
            if self.oov_handling == "raise":
                return {"skip": True, "error": f"'{identifier}': {e}"}
            return _skip(identifier, str(e))
        if not data["paths"].any():
            return _skip(identifier, "No valid token sequence")
        original_count = sum(len(reading.paths) for word in g2p_words for reading in word.readings)
        dropped = original_count - int(data["candidates"].sum())
        warning = f"Dropped {dropped} OOV pronunciation(s) in '{identifier}'" if dropped else ""

        audio, sr = load_audio(audio_path)
        if sr != self.sample_rate:
            audio = librosa.resample(
                audio, orig_sr=sr, target_sr=self.sample_rate,
            )

        return {
            "skip": False,
            "identifier": identifier,
            "warning": warning,
            "waveform": torch.from_numpy(audio).float(),
            "duration": len(audio) / self.sample_rate,
            **{key: torch.from_numpy(value) for key, value in data.items()},
            "lexicon": lexicon,
            "texts": texts,
        }

    @staticmethod
    def collate(batch: list[dict[str, Any]]) -> dict[str, Any]:
        errors = [err for b in batch if (err := b.get("error"))]
        if errors:
            raise RuntimeError("\n".join(errors))

        valid = [b for b in batch if not b["skip"]]
        warning = [warn for b in batch if (warn := b.get("warning"))]

        if not valid:
            return {"warning": warning}

        return {
            "warning": warning,
            "identifier": [b["identifier"] for b in valid],
            "waveform": collate_nd(
                [b["waveform"] for b in valid], pad_value=0.,
            ),
            "duration": torch.tensor(
                [b["duration"] for b in valid], dtype=torch.float32,
            ),
            **{
                key: collate_nd([b[key] for b in valid], ndim=ndim)
                for key, ndim in (("paths", 2), ("words", 1), ("groups", 2), ("candidates", 2))
            },
            "lexicon": [b["lexicon"] for b in valid],
            "texts": [b["texts"] for b in valid],
        }


class TextGridDataset(torch.utils.data.Dataset):
    """Parses TextGrid files from a directory recursively.

    Each item returns intervals from a named tier with empty marks filtered.
    """

    def __init__(
        self,
        directory: pathlib.Path,
        tier_name: str = "phones",
        stop_symbols: set[str] | None = None,
    ):
        self.directory = pathlib.Path(directory)
        if not self.directory.is_dir():
            raise FileNotFoundError(f"Directory not found: {directory}")
        self.tier_name = tier_name
        self.stop_symbols = stop_symbols or set()
        self._files = sorted(
            p for p in self.directory.glob("**/*.TextGrid")
            if p.is_file()
        )
        if not self._files:
            raise FileNotFoundError(
                f"No .TextGrid files found in {directory}"
            )

    def __len__(self):
        return len(self._files)

    def __getitem__(self, index):
        filepath = self._files[index]
        identifier = str(
            filepath.relative_to(self.directory).with_suffix("")
        )
        tg = textgrid.TextGrid.fromFile(str(filepath))

        onsets: list[float] = []
        offsets: list[float] = []
        marks: list[str] = []

        for tier in tg:
            if tier.name == self.tier_name and isinstance(
                tier, textgrid.IntervalTier
            ):
                for interval in tier:
                    mark = interval.mark.strip()
                    if not mark or mark in self.stop_symbols:
                        continue
                    onsets.append(float(interval.minTime))
                    offsets.append(float(interval.maxTime))
                    marks.append(mark)
                break

        return {
            "identifier": identifier,
            "onsets": onsets,
            "offsets": offsets,
            "marks": marks,
        }


class PairedDataset(torch.utils.data.Dataset):
    """Pairs two datasets by identifier for offline evaluation.

    Accepts any two Datasets whose items contain ``"identifier"``,
    ``"onsets"``, ``"offsets"``, ``"marks"``.  Verifies label equality,
    builds an internal vocabulary, and encodes spans + tokens.
    """

    def __init__(
        self,
        pred_dataset: torch.utils.data.Dataset,
        gt_dataset: torch.utils.data.Dataset,
        mismatch_handling: Literal["raise", "skip"] = "raise",
    ):
        self.mismatch_handling = mismatch_handling
        self.vocab_size: int = 0

        pred_by_id = {item["identifier"]: item for item in pred_dataset}
        gt_by_id = {item["identifier"]: item for item in gt_dataset}

        pred_only = set(pred_by_id) - set(gt_by_id)
        gt_only = set(gt_by_id) - set(pred_by_id)
        for oid in sorted(pred_only):
            logging.warning(f"Prediction '{oid}' has no ground-truth counterpart, skipped")
        for oid in sorted(gt_only):
            logging.warning(f"Ground-truth '{oid}' has no prediction counterpart, skipped")

        common = sorted(set(pred_by_id) & set(gt_by_id))
        if not common:
            raise ValueError(
                "No matching identifiers found between pred and gt datasets"
            )

        self._items: list[dict] = []
        all_marks: set[str] = set()

        for identifier in common:
            P = pred_by_id[identifier]
            G = gt_by_id[identifier]

            p_marks = P["marks"]
            g_marks = G["marks"]

            if len(p_marks) != len(g_marks):
                msg = (
                    f"'{identifier}': pred has {len(p_marks)} intervals, "
                    f"gt has {len(g_marks)}"
                )
                if self.mismatch_handling == "raise":
                    raise ValueError(msg)
                logging.warning(f"Skipping {msg}")
                continue

            for i, (mp, mg) in enumerate(zip(p_marks, g_marks)):
                if mp != mg:
                    msg = (
                        f"'{identifier}'[{i}]: pred='{mp}' vs gt='{mg}'"
                    )
                    if self.mismatch_handling == "raise":
                        raise ValueError(msg)
                    logging.warning(f"Skipping {msg}")
                    break
            else:
                all_marks.update(p_marks)
                self._items.append({
                    "identifier": identifier,
                    "pred_onsets": P["onsets"],
                    "pred_offsets": P["offsets"],
                    "gt_onsets": G["onsets"],
                    "gt_offsets": G["offsets"],
                    "marks": p_marks,
                })

        if not self._items:
            raise ValueError("No valid pairs after verification")

        self._mark_to_id = {
            label: i + 3  # NUM_RESERVED_TOKENS
            for i, label in enumerate(sorted(all_marks))
        }
        self.vocab = Vocabulary(symbol_to_id=self._mark_to_id)

    def __len__(self):
        return len(self._items)

    def __getitem__(self, index):
        item = self._items[index]
        tokens = torch.tensor(
            [self._mark_to_id[m] for m in item["marks"]], dtype=torch.int64
        )
        spans_pred = torch.stack([
            torch.tensor(item["pred_onsets"], dtype=torch.float32),
            torch.tensor(item["pred_offsets"], dtype=torch.float32),
        ], dim=-1)
        spans_gt = torch.stack([
            torch.tensor(item["gt_onsets"], dtype=torch.float32),
            torch.tensor(item["gt_offsets"], dtype=torch.float32),
        ], dim=-1)
        return {
            "identifier": item["identifier"],
            "tokens": tokens,
            "spans": spans_gt,
            "spans_pred": spans_pred,
        }

    @staticmethod
    def collate(samples: list[dict]) -> dict:
        return {
            "identifier": [s["identifier"] for s in samples],
            "tokens": collate_nd([s["tokens"] for s in samples], pad_value=0, ndim=1),
            "spans": collate_nd([s["spans"] for s in samples], pad_value=0.0, ndim=2),
            "spans_pred": collate_nd(
                [s["spans_pred"] for s in samples], pad_value=0.0, ndim=2
            ),
        }
