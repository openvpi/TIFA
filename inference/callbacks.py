import json
import pathlib
import re
from itertools import groupby
from typing import Any

import lightning.pytorch.callbacks
import matplotlib.pyplot as plt
import numpy as np
import textgrid
import torch
from g2pflow import G2PGroup, G2PReading, G2PWord, to_pfml
from lightning_utilities.core.rank_zero import rank_zero_only
from torch import nn

from lib import logging
from lib.plot import (
    alignment_to_figure,
    emission_to_figure,
    metric_histogram_figure,
    metric_scatter_figure,
    topk_bar_figure,
)
from lib.vocabulary import Vocabulary
from modules.metrics import (
    BoundaryErrorRate,
    BoundaryMAE,
    Confidence,
    Determinacy,
    Monotonicity,
    OverlapRatioCollection,
    PairConjunctionMAE,
    compute_boundary_mae,
    compute_confidence,
    compute_determinacy,
    compute_monotonicity,
    compute_overlap,
)


class _CompactDict(dict):
    """Dict that serializes as a compact single-line JSON object."""


class _CompactEncoder(json.JSONEncoder):
    """Encoder that serializes ``_CompactDict`` without indentation."""

    _SENTINEL = "<<compact>>"

    def encode(self, o):
        o = self._mark_compact(o)
        result = super().encode(o)
        esc = re.escape(self._SENTINEL)
        pattern = re.compile(rf'"{esc}(.*?){esc}"', re.DOTALL)

        def _replace(m):
            inner = m.group(1)
            inner = inner.replace('\\"', '"')
            return inner

        return pattern.sub(_replace, result)

    @classmethod
    def _mark_compact(cls, o):
        s = cls._SENTINEL
        if isinstance(o, _CompactDict):
            return s + json.dumps(dict(o), ensure_ascii=False) + s
        if isinstance(o, dict):
            return {k: cls._mark_compact(v) for k, v in o.items()}
        if isinstance(o, list):
            return [cls._mark_compact(item) for item in o]
        return o


def _preserve_skipped_spans(
    spans: list[list[float]], groups: list[int], total_duration: float,
) -> tuple[list[tuple[float, float]], float]:
    """Place 1ms skipped intervals inside permitted gaps, then repair overlaps.

    Inputs have already been rounded to milliseconds. Work on a copy in
    integer milliseconds so output-only repairs cannot change decoded spans.
    """
    original = [(int(round(start * 1000)), int(round(end * 1000))) for start, end in spans]
    duration_ms = int(round(total_duration * 1000))
    intervals = []
    N = len(original)
    i = 0
    while i < N:
        left = intervals[-1][1] if intervals else 0
        onset, offset = original[i]
        if offset > onset:
            onset = max(onset, left)
            intervals.append((onset, max(offset, onset + 1)))
            i += 1
            continue

        end = i + 1
        while end < N and original[end][0] >= original[end][1]:
            end += 1
        right = max(left, original[end][0] if end < N else duration_ms)
        gap_index = i
        while gap_index <= end:
            if gap_index == 0 or gap_index == N or groups[gap_index - 1] != groups[gap_index]:
                break
            gap_index += 1
        left_count = min(gap_index, end) - i
        right_count = end - i - left_count
        for k in range(left_count):
            intervals.append((left + k, left + k + 1))
        right_start = max(left + left_count, right - right_count)
        for k in range(right_count):
            intervals.append((right_start + k, right_start + k + 1))
        i = end

    if intervals:
        duration_ms = max(duration_ms, intervals[-1][1])
    return [(start / 1000, end / 1000) for start, end in intervals], duration_ms / 1000


class SaveTextGridCallback(lightning.pytorch.callbacks.Callback):
    """Writes 3-tier TextGrid files and optional PFML from alignment results.

    Tiers:
    - texts: semantic word intervals with G2P word text
    - words: intervals from decoded spans, labels from G2P output
    - phones: intervals from decoded spans, labels from G2P output

    Expects spans and duration in frames; converts to seconds via
    ``timestep``. PFML retains full phoneme names after skip handling.
    """

    def __init__(
            self,
            output_dir: str | pathlib.Path,
            language: str | None = None,
            timestep: float = 1.0,
            skip_handling: str = "omit",
            save_pfml: bool = False,
    ):
        super().__init__()
        self.output_dir = pathlib.Path(output_dir)
        self.language = language
        self.timestep = timestep
        self.skip_handling = skip_handling
        self.save_pfml = save_pfml

    def on_predict_batch_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            outputs: list[dict[str, Any]],
            batch: dict[str, Any],
            *args, **kwargs,
    ) -> None:
        timestep = self.timestep
        for result in outputs:
            identifier = result["identifier"]
            words = result["words"].tolist()  # Semantic word IDs, 1-based
            groups = result["groups"].tolist()  # Selected pronunciation groups, 1-based
            spans = (result["spans"].float() * timestep).tolist()  # frames -> seconds
            phonemes = result["phonemes"]
            pfml_phonemes = phonemes
            if self.language:
                prefix = f"{self.language}/"
                phonemes = [
                    ph[len(prefix):] if ph.startswith(prefix) else ph
                    for ph in phonemes
                ]
            group_scripts = []
            for candidates, choice in zip(result["lexicon"], result["choices"].tolist()):
                if choice > 0:
                    group_scripts.extend(candidates[choice - 1]["scripts"])
            N = len(spans)
            if N == 0:
                continue

            # Step 1: round to 3 decimals
            total_duration = round(result["spectrogram"].shape[0] * timestep, 3)
            for s in spans:
                s[0] = round(s[0], 3)
                s[1] = round(s[1], 3)

            # Step 2: identify and warn on zero-width spans
            eps = 0.001
            zero_idx = [i for i in range(N) if spans[i][0] >= spans[i][1]]
            for i in zero_idx:
                onset, offset = spans[i]
                logging.warning(
                    f"Skipped state encountered in '{identifier}', idx {i}: {phonemes[i]} [{onset}, {offset}]",
                    callback=trainer.progress_bar_callback.print,
                )

            # Step 3: handle zero-width spans based on mode
            if self.skip_handling == "discard":
                if zero_idx:
                    continue

            elif self.skip_handling == "omit":
                if zero_idx:
                    keep = [i for i in range(N) if i not in zero_idx]
                    if not keep:
                        continue
                    spans = [spans[i] for i in keep]
                    phonemes = [phonemes[i] for i in keep]
                    if self.save_pfml:
                        pfml_phonemes = [pfml_phonemes[i] for i in keep]
                    groups = [groups[i] for i in keep]
                    words = [words[i] for i in keep]
                    N = len(spans)

            else:  # preserve
                spans, total_duration = _preserve_skipped_spans(spans, groups, total_duration)

            # Step 4: prebuild phone intervals
            phone_intervals = []
            for n in range(N):
                onset, offset = spans[n]
                if phone_intervals and onset < phone_intervals[-1][1]:
                    onset = phone_intervals[-1][1]
                if offset <= onset:
                    offset = onset + eps
                phone_intervals.append((onset, offset, phonemes[n]))
            if phone_intervals[-1][1] > total_duration:
                total_duration = phone_intervals[-1][1]

            # Step 5: aggregate semantic words and pronunciation groups.
            # IDs stay stable when zero-width phones are omitted.
            tg = textgrid.TextGrid()
            for name, owners, labels in (
                ("texts", words, result["texts"]),
                ("words", groups, group_scripts),
            ):
                intervals = []
                i = 0
                while i < N:
                    owner = owners[i]
                    j = i + 1
                    while j < N and owners[j] == owner:
                        j += 1
                    onset = phone_intervals[i][0]
                    offset = phone_intervals[j - 1][1]
                    if intervals and onset < intervals[-1][1]:
                        onset = intervals[-1][1]
                    if offset <= onset:
                        offset = onset + eps
                    intervals.append((onset, offset, labels[owner - 1]))
                    i = j
                tier = textgrid.IntervalTier(name, 0, total_duration)
                for onset, offset, label in intervals:
                    tier.add(onset, offset, label)
                tg.append(tier)

            # Step 6: append phones and write the output files.
            phones_tier = textgrid.IntervalTier("phones", 0, total_duration)
            for onset, offset, label in phone_intervals:
                phones_tier.add(onset, offset, label)
            tg.append(phones_tier)

            output_path = self.output_dir / f"{identifier}.TextGrid"
            output_path.parent.mkdir(parents=True, exist_ok=True)
            if self.save_pfml:
                pfml_words = []
                for word_id, word_items in groupby(
                    zip(words, groups, pfml_phonemes), key=lambda item: item[0],
                ):
                    path = [
                        G2PGroup(
                            script=group_scripts[group_id - 1],
                            phonemes=[phone for _, _, phone in group_items],
                        )
                        for group_id, group_items in groupby(word_items, key=lambda item: item[1])
                    ]
                    pfml_words.append(G2PWord(
                        text=result["texts"][word_id - 1],
                        readings=[G2PReading(paths=[path])],
                    ))
                pfml_source = to_pfml(pfml_words)
                with output_path.with_suffix(".pfml").open("w", encoding="utf8") as f:
                    f.write(pfml_source + "\n")
            tg.write(str(output_path))


class SavePlotCallback(lightning.pytorch.callbacks.Callback):
    """Saves per-sample similarity and alignment plots.

    In predict mode: similarity + pred-only alignment plots, using
    phoneme labels from G2P output.

    In test mode: similarity + alignment with GT overlay, using
    vocab-decoded token labels.
    """

    def __init__(
            self,
            output_dir: pathlib.Path,
            vocab: "Vocabulary" = None,
            identifiers: list = None,
    ):
        super().__init__()
        self.output_dir = pathlib.Path(output_dir)
        self.vocab = vocab
        self._num_digits = len(str(len(identifiers))) if identifiers else 0
        self._identifiers = identifiers

    def on_predict_batch_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            outputs: list[dict[str, Any]],
            batch: dict[str, Any],
            *args, **kwargs,
    ) -> None:
        for result in outputs:
            sim = result["similarity"]  # [T_i, N_i]
            identifier = result["identifier"]
            phonemes = result["phonemes"]
            N_i = sim.shape[1]

            # Similarity plot
            fig_sim = emission_to_figure(
                sim.T.float().detach().cpu().numpy(),
                regions=None,
                title=identifier,
                token_labels=phonemes,
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            fig_sim.savefig(self.output_dir / f"{identifier}_sim.jpg")
            plt.close(fig_sim)

            # Alignment plot (pred-only, no GT)
            spec = result.get("spectrogram")
            spans = result["spans"]  # [N_i, 2] in frames
            T_i = sim.shape[0]
            if spec is not None and T_i > 0 and N_i > 0:
                fig_align = alignment_to_figure(
                    spec.detach().cpu().numpy(),
                    token_labels=phonemes[:N_i] if phonemes else None,
                    pred_spans=spans.detach().cpu().numpy(),
                    title=identifier,
                )
                fig_align.savefig(self.output_dir / f"{identifier}_align.jpg")
                plt.close(fig_align)

    def on_test_batch_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            outputs: dict,
            batch: dict,
            *args, **kwargs,
    ) -> None:
        spectrogram = batch["spectrogram"]
        similarity = outputs["similarity"]
        regions = batch["regions"]
        tokens = batch["tokens"]
        spans_gt = batch["spans"]
        spans_pred = outputs["spans"]
        indices = batch["indices"]
        T_all = batch["T"]
        N_all = batch["N"]

        B = indices.shape[0]
        for i in range(B):
            T_i = int(T_all[i].item())
            N_i = int(N_all[i].item())
            if T_i == 0 or N_i == 0:
                continue

            data_idx = int(indices[i].item())
            token_ids = tokens[i, :N_i].tolist()
            token_labels = [
                self.vocab.decode(int(tid), stringfy=True) or str(tid)
                for tid in token_ids
            ]

            item_path = self._identifiers[data_idx]
            name = str(data_idx).zfill(self._num_digits)

            # Similarity plot
            sim = similarity[i, :T_i, :N_i].float().detach().cpu().numpy()
            fig_sim = emission_to_figure(
                sim.T,
                regions=regions[i, :T_i].detach().cpu().numpy(),
                title=item_path,
                token_labels=token_labels,
            )
            self.output_dir.mkdir(parents=True, exist_ok=True)
            fig_sim.savefig(self.output_dir / f"{name}_sim.jpg")
            plt.close(fig_sim)

            # Alignment plot
            spec = spectrogram[i, :T_i].detach().cpu().numpy()
            ps = spans_pred[i, :N_i].detach().cpu().numpy()
            gs = spans_gt[i, :N_i].detach().cpu().numpy()
            fig_align = alignment_to_figure(
                spec,
                token_labels=token_labels,
                pred_spans=ps,
                gt_spans=gs,
                title=item_path,
            )
            fig_align.savefig(self.output_dir / f"{name}_align.jpg")
            plt.close(fig_align)


def _gather_records(records):
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        return records
    world_size = torch.distributed.get_world_size()
    if world_size <= 1:
        return records
    gathered = [None] * world_size
    torch.distributed.all_gather_object(gathered, records)
    merged = []
    for r in gathered:
        merged.extend(r)
    return merged


class StatisticsCallback(lightning.pytorch.callbacks.Callback):
    """Accumulates per-sample determinacy/confidence scores, saves JSON and
    statistic plots (histograms + scatter).
    """

    def __init__(
            self,
            save_dir: pathlib.Path,
            determinacy_power: float = 2.0,
            determinacy_width: int | None = 5,
            monotonicity_power: float = 2.0,
            monotonicity_width: int | None = None,
            identifiers: list[str] | None = None,
            unit_factor: float = 1.0,
    ):
        super().__init__()
        self.save_dir = pathlib.Path(save_dir)
        self.power = determinacy_power
        self.width = determinacy_width
        self.monotonicity_power = monotonicity_power
        self.monotonicity_width = monotonicity_width
        self._identifiers = identifiers
        self._unit_factor = unit_factor
        self._metric_records: list[dict] = []
        self._score_records: list[dict] = []
        self._saved = False

    def on_test_batch_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            outputs: dict,
            batch: dict,
            *args, **kwargs,
    ) -> None:
        spans_pred = outputs["spans"].float() * self._unit_factor  # [B, N_max, 2]
        spans_gt = batch["spans"].float() * self._unit_factor  # [B, N_max, 2]
        tokens = batch["tokens"]  # [B, N_max]

        B = tokens.shape[0]
        device = tokens.device

        onset_error, onset_count = compute_boundary_mae(
            spans_pred, spans_gt, tokens, "onset",
        )  # [B] each
        offset_error, offset_count = compute_boundary_mae(
            spans_pred, spans_gt, tokens, "offset",
        )  # [B] each
        overlap_sum, pred_sum, gt_sum = compute_overlap(
            spans_pred, spans_gt, tokens,
        )  # [B] each

        has_similarity = "similarity" in outputs
        if has_similarity:
            similarity = outputs["similarity"].float()  # [B, T_max, N_max]
        else:
            similarity = None

        has_indices = "indices" in batch
        has_identifiers = "identifier" in batch

        for i in range(B):
            N_i = int((tokens[i] != 0).sum().item())
            if N_i == 0:
                continue

            record: dict[str, object] = {}
            if has_indices:
                data_idx = int(batch["indices"][i].item())
                record["index"] = data_idx
                record["identifier"] = self._identifiers[data_idx] if self._identifiers else str(data_idx)
            elif has_identifiers:
                record["identifier"] = batch["identifier"][i]
            else:
                record["identifier"] = str(i)
            record["b_mae_onset"] = (onset_error[i] / onset_count[i].clamp(min=1)).item()
            record["b_mae_offset"] = (offset_error[i] / offset_count[i].clamp(min=1)).item()
            record["overlap_precision"] = (overlap_sum[i] / (pred_sum[i] + 1e-6)).item()
            record["overlap_recall"] = (overlap_sum[i] / (gt_sum[i] + 1e-6)).item()
            record["num_tokens"] = N_i
            record["num_skipped_tokens"] = int((
                (spans_pred[i, :, 0] == spans_pred[i, :, 1]) & (tokens[i] != 0)
            ).sum().item())
            if "T" in batch:
                record["num_frames"] = int(batch["T"][i].item())

            if has_similarity:
                sim_i = similarity[i, :, :N_i]
                span_i = spans_pred[i, :N_i]
                T_sim = int(sim_i.shape[0])
                if T_sim > 0:
                    t_mask = torch.ones(T_sim, dtype=torch.bool, device=device)
                    n_mask = tokens[i, :N_i] != 0

                    record["confidence"] = compute_confidence(
                        span_i, sim_i, t_mask, n_mask,
                    ).item()

                    num, denom = compute_determinacy(
                        span_i, sim_i, t_mask, n_mask,
                        power=self.power,
                        width=self.width,
                    )
                    record["determinacy"] = (num / (denom + 1e-8)).item()

                    record["monotonicity"] = compute_monotonicity(
                        span_i, sim_i, t_mask, n_mask,
                        power=self.monotonicity_power,
                        width=self.monotonicity_width,
                    ).item()

            self._metric_records.append(record)

    def on_predict_batch_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            outputs: list[dict],
            batch: dict,
            *args, **kwargs,
    ) -> None:
        for result in outputs:
            if "similarity" not in result:
                continue

            spans_frames = result["spans"].float()  # [N_i, 2] in frames
            similarity = result["similarity"].float()  # [T_i, N_i]
            tokens = result["tokens"]  # [N_i]
            N_i = tokens.shape[0]
            T_i = similarity.shape[0]

            if T_i == 0 or N_i == 0:
                continue

            device = similarity.device

            t_mask = torch.ones(T_i, dtype=torch.bool, device=device)
            n_mask = (tokens[:N_i] != 0).to(device)

            confidence = compute_confidence(
                spans_frames,
                similarity[:T_i, :N_i],
                t_mask,
                n_mask,
            ).item()

            num, denom = compute_determinacy(
                spans_frames,
                similarity[:T_i, :N_i],
                t_mask,
                n_mask,
                power=self.power,
                width=self.width,
            )
            determinacy = (num / (denom + 1e-8)).item()

            monotonicity = compute_monotonicity(
                spans_frames,
                similarity[:T_i, :N_i],
                t_mask,
                n_mask,
                power=self.monotonicity_power,
                width=self.monotonicity_width,
            ).item()

            self._metric_records.append({
                "identifier": result["identifier"],
                "agreement": result["agreement"],
                "confidence": confidence,
                "determinacy": determinacy,
                "monotonicity": monotonicity,
                "num_frames": T_i,
                "num_tokens": N_i,
                "num_skipped_tokens": int((
                    (spans_frames[:, 0] == spans_frames[:, 1]) & n_mask
                ).sum().item()),
            })

            if result.get("scores") is not None:
                words = []
                for w, (text, candidates, chosen) in enumerate(zip(
                    result["texts"], result["lexicon"], result["choices"].tolist(),
                )):
                    width = len(candidates)
                    if width <= 1:
                        continue
                    words.append({
                        "index": w,
                        "text": text,
                        "chosen": chosen - 1,
                        "alternatives": [
                            _CompactDict({
                                "index": c,
                                "scripts": candidate["scripts"],
                                "phones": candidate["phonemes"],
                                "score": float(result["scores"][w, c].item()),
                            })
                            for c, candidate in enumerate(candidates)
                        ],
                    })
                if words:
                    self._score_records.append({
                        "identifier": result["identifier"],
                        "words": words,
                    })

    def on_test_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            *args, **kwargs,
    ) -> None:
        self._save()

    def on_predict_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            *args, **kwargs,
    ) -> None:
        self._save()

    def _save(self) -> None:
        if self._saved:
            return
        self._saved = True

        records = _gather_records(self._metric_records)
        score_records = _gather_records(self._score_records)

        if not records:
            return

        if (
                not torch.distributed.is_available()
                or not torch.distributed.is_initialized()
                or torch.distributed.get_rank() == 0
        ):
            # Sort direction: ascending=True means low=bad, ascending=False means high=bad
            _METRIC_DIRECTIONS: list[tuple[str, bool]] = [
                ("b_mae_onset", False),
                ("b_mae_offset", False),
                ("overlap_precision", True),
                ("overlap_recall", True),
                ("agreement", True),
                ("confidence", True),
                ("determinacy", True),
                ("monotonicity", True),
            ]
            available = [
                (key, asc) for key, asc in _METRIC_DIRECTIONS
                if key in records[0]
            ]
            ranks: dict[str, dict[str, int]] = {}
            for key, ascending in available:
                sorted_recs = sorted(
                    records, key=lambda rc: rc[key], reverse=not ascending,
                )
                ranks[key] = {
                    rc["identifier"]: i for i, rc in enumerate(sorted_recs)
                }

            def _sort_key(rc):
                identifier = rc["identifier"]
                _r = tuple(ranks[key][identifier] for key, _ in available)
                return (-rc["num_skipped_tokens"], min(_r)) + _r

            records.sort(key=_sort_key)

            self.save_dir.mkdir(parents=True, exist_ok=True)
            with open(self.save_dir / "diagnosis.json", "w", encoding="utf8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)
            self._save_stat_plots(records)

            with open(self.save_dir / "scores.json", "w", encoding="utf8") as f:
                f.write(_CompactEncoder(
                    indent=2, ensure_ascii=False,
                ).encode(score_records))

    def _save_stat_plots(self, records: list[dict]) -> None:
        if not records:
            return

        has_num_frames = "num_frames" in records[0]
        plot_keys: list[str] = []
        for key in (
                "b_mae_onset", "b_mae_offset",
                "overlap_precision", "overlap_recall",
                "agreement", "confidence", "determinacy", "monotonicity",
        ):
            if key in records[0]:
                plot_keys.append(key)

        m: dict[str, np.ndarray] = {}
        for key in plot_keys:
            m[key] = np.array([r[key] for r in records])
        if has_num_frames:
            m["num_frames"] = np.array(
                [r["num_frames"] for r in records], dtype=np.int32,
            )

        _LOG_METRICS = {"b_mae_onset", "b_mae_offset"}

        for key in plot_keys:
            use_log = key in _LOG_METRICS
            fig = metric_histogram_figure(
                m[key], label=key, log_x=use_log, log_x_min=0.1,
            )
            safe_key = key.replace("/", "_")
            fig.savefig(
                self.save_dir / f"{safe_key}_histogram.jpg",
                bbox_inches="tight",
            )
            plt.close(fig)
            if has_num_frames:
                fig = metric_scatter_figure(
                    m["num_frames"], m[key],
                    xlabel="Number of Frames", ylabel=key,
                    title=key, log_y=use_log, log_y_min=0.1,
                )
                fig.savefig(
                    self.save_dir / f"{safe_key}_scatter.jpg",
                    bbox_inches="tight",
                )
                plt.close(fig)


class EvaluationMetricsCallback(lightning.pytorch.callbacks.Callback):
    """Owns metric instances, converts spans, computes + exports JSON.

    ``unit`` is ``"frame"`` or ``"ms"`` and determines the conversion
    factor applied to raw spans before feeding metrics.

    ``vocab`` provides ``.vocab_size`` and ``.decode(token_id)`` for
    per-token statistics.
    """

    def __init__(
            self,
            unit: str,
            vocab: "Vocabulary",
            save_path: pathlib.Path,
            ber_tols: list[int] | None = None,
            token_topk: list[int] | None = None,
            pair_topk: list[int] | None = None,
            determinacy_power: float = 2.0,
            determinacy_width: int | None = 5,
            monotonicity_power: float = 2.0,
            monotonicity_width: int | None = None,
    ):
        super().__init__()
        if unit == "frame":
            self._unit_factor = 1.0
        elif unit == "ms":
            self._unit_factor = 1000.0
        else:
            raise ValueError(f"Unknown unit: {unit}")
        self.unit = unit
        self.save_path = pathlib.Path(save_path)
        self.ber_tols = ber_tols or []
        self.token_topk = token_topk or []
        self.pair_topk = pair_topk or []
        self._vocab = vocab
        self._similarity_seen = False
        self.determinacy_power = determinacy_power
        self.determinacy_width = determinacy_width
        self.monotonicity_power = monotonicity_power
        self.monotonicity_width = monotonicity_width
        self._results: dict | None = None

        metrics: dict[str, nn.Module] = {}
        max_k = max(token_topk) if token_topk else 0
        for tol in ber_tols:
            for mode in ("onset", "offset", "both"):
                metrics[f"ber/{mode}/{tol}"] = BoundaryErrorRate(
                    tolerance=tol, mode=mode,
                )
            if max_k:
                for mode in ("onset", "offset"):
                    metrics[f"ber/{mode}/{tol}/{max_k}"] = BoundaryErrorRate(
                        tolerance=tol, mode=mode,
                        vocab_size=self._vocab.vocab_size, k=max_k,
                    )
        for k in token_topk:
            for mode in ("onset", "offset"):
                metrics[f"b_mae/{mode}/{k}"] = BoundaryMAE(
                    mode=mode, vocab_size=self._vocab.vocab_size, k=k,
                )
            metrics[f"overlap/{k}"] = OverlapRatioCollection(
                template=f"overlap_{{}}@{k}", vocab_size=self._vocab.vocab_size, k=k,
            )
        for k in pair_topk:
            metrics[f"conj_mae/{k}"] = PairConjunctionMAE(
                vocab_size=self._vocab.vocab_size, k=k,
            )
        self.confidence = Confidence()
        self.determinacy = Determinacy(
            power=determinacy_power, width=determinacy_width,
        )
        self.monotonicity = Monotonicity(
            power=monotonicity_power, width=monotonicity_width,
        )
        self.metrics = nn.ModuleDict(metrics)

    def on_test_start(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            *args, **kwargs,
    ) -> None:
        self.metrics.to(trainer.strategy.root_device)
        self.confidence.to(trainer.strategy.root_device)
        self.determinacy.to(trainer.strategy.root_device)
        self.monotonicity.to(trainer.strategy.root_device)

    def on_test_batch_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            outputs: dict,
            batch: dict,
            *args, **kwargs,
    ) -> None:
        spans_pred_ms = outputs["spans"].float() * self._unit_factor
        spans_gt_ms = batch["spans"].float() * self._unit_factor
        tokens = batch["tokens"]
        for metric in self.metrics.values():
            metric.update(spans_pred_ms, spans_gt_ms, tokens)

        if "similarity" in outputs:
            T_max = outputs["similarity"].shape[1]
            device = batch["T"].device
            t_mask = torch.arange(T_max, device=device).unsqueeze(0) < batch["T"].unsqueeze(1)
            n_mask = tokens != 0
            self.confidence.update(
                outputs["spans"],
                outputs["similarity"],
                t_mask,
                n_mask,
            )
            self.determinacy.update(
                outputs["spans"],
                outputs["similarity"],
                t_mask,
                n_mask,
            )
            self.monotonicity.update(
                outputs["spans"],
                outputs["similarity"],
                t_mask,
                n_mask,
            )
            self._similarity_seen = True

    def on_test_end(
            self,
            trainer: lightning.pytorch.Trainer,
            pl_module: lightning.pytorch.LightningModule,
            *args, **kwargs,
    ) -> None:
        self._results = self._build_summary()

        @rank_zero_only
        def _save_outputs():
            self.save_path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.save_path, "w", encoding="utf8") as f:
                json.dump(self._results, f, indent=2)
            self._save_statistic_plots()

        _save_outputs()

    @property
    def results(self) -> dict | None:
        return self._results

    # ------------------------------------------------------------------
    # Summary builder
    # ------------------------------------------------------------------

    def _build_summary(self) -> dict:
        metrics_list: list[dict] = []

        # -- BER --
        ber_variants: list[dict] = []
        ber_statistics: list[dict] = []
        for tol in self.ber_tols:
            for mode in ("onset", "offset", "both"):
                ber_variants.append({
                    "arguments": {"mode": mode, "tolerance": tol},
                    "value": float(self.metrics[f"ber/{mode}/{tol}"].compute().item()),
                })
            if self._vocab and self.token_topk:
                max_k = max(self.token_topk)
                for mode in ("onset", "offset"):
                    m = self.metrics[f"ber/{mode}/{tol}/{max_k}"]
                    ber_variants.append({
                        "arguments": {"mode": mode, "tolerance": tol, "k": max_k},
                        "value": float(m.compute().item()),
                    })
                    top = m.compute_top_k()
                    if top:
                        ber_statistics.append({
                            "arguments": {"mode": mode, "tolerance": tol},
                            "groups": [
                                {"key": self._vocab.decode(tid, stringfy=True), "value": float(v.item())}
                                for tid, v in sorted(top.items(), key=lambda x: -x[1])
                            ],
                        })
        metrics_list.append({
            "name": "BER",
            "variants": ber_variants, "statistics": ber_statistics,
        })

        # -- B-MAE --
        b_mae_variants: list[dict] = []
        b_mae_statistics: list[dict] = []
        for mode in ("onset", "offset"):
            for k in self.token_topk:
                b_mae_variants.append({
                    "arguments": {"mode": mode, "k": k},
                    "value": float(self.metrics[f"b_mae/{mode}/{k}"].compute().item()),
                })
            if self._vocab:
                max_k = max(self.token_topk)
                m = self.metrics[f"b_mae/{mode}/{max_k}"]
                top = m.compute_top_k()
                if top:
                    b_mae_statistics.append({
                        "arguments": {"mode": mode},
                        "groups": [
                            {"key": self._vocab.decode(tid, stringfy=True), "value": float(v.item())}
                            for tid, v in sorted(top.items(), key=lambda x: -x[1])
                        ],
                    })
        metrics_list.append({
            "name": "B-MAE", "unit": self.unit,
            "variants": b_mae_variants, "statistics": b_mae_statistics,
        })

        # -- Overlap --
        ov_variants: list[dict] = []
        ov_statistics: list[dict] = []
        for k in self.token_topk:
            ov_metric = self.metrics[f"overlap/{k}"]
            vals = ov_metric.compute()
            ov_variants.append({
                "arguments": {"k": k},
                "value": {
                    "precision": float(vals[f"overlap_precision@{k}"].item()),
                    "recall": float(vals[f"overlap_recall@{k}"].item()),
                },
            })
        if self._vocab and self.token_topk:
            max_k = max(self.token_topk)
            ov_metric = self.metrics[f"overlap/{max_k}"]
            top = ov_metric.compute_top_k()
            if top:
                for metric_name in (f"overlap_precision@{max_k}", f"overlap_recall@{max_k}"):
                    if metric_name in top and top[metric_name]:
                        ov_statistics.append({
                            "arguments": {"k": max_k, "metric": metric_name.split("_", 1)[1].split("@")[0]},
                            "groups": [
                                {"key": self._vocab.decode(tid, stringfy=True), "value": float(v.item())}
                                for tid, v in sorted(
                                    top[metric_name].items(), key=lambda x: -x[1],
                                )
                            ],
                        })
        metrics_list.append({
            "name": "Overlap",
            "variants": ov_variants, "statistics": ov_statistics,
        })

        # -- Conj-MAE --
        cm_variants: list[dict] = []
        cm_statistics: list[dict] = []
        for k in self.pair_topk:
            cm = self.metrics[f"conj_mae/{k}"]
            cm_variants.append({
                "arguments": {"k": k},
                "value": float(cm.compute().item()),
            })
        if self._vocab and self.pair_topk:
            max_k = max(self.pair_topk)
            cm = self.metrics[f"conj_mae/{max_k}"]
            top = cm.compute_top_k()
            if top:
                cm_statistics.append({
                    "arguments": {},
                    "groups": [
                        {
                            "key": f"{self._vocab.decode(ti, stringfy=True)},{self._vocab.decode(tj, stringfy=True)}",
                            "value": float(v.item()),
                        }
                        for (ti, tj), v in sorted(
                            top.items(), key=lambda x: -x[1],
                        )
                    ],
                })
        metrics_list.append({
            "name": "Conj-MAE", "unit": self.unit,
            "variants": cm_variants, "statistics": cm_statistics,
        })

        # Confidence and Determinacy
        if self._similarity_seen:
            conf_variants = [{
                "arguments": {},
                "value": float(self.confidence.compute().item()),
            }]
            metrics_list.append({
                "name": "Confidence",
                "variants": conf_variants,
            })

            det_variants = [{
                "arguments": {"power": self.determinacy_power, "width": self.determinacy_width},
                "value": float(self.determinacy.compute().item()),
            }]
            metrics_list.append({
                "name": "Determinacy",
                "variants": det_variants,
            })

            mono_variants = [{
                "arguments": {"power": self.monotonicity_power, "width": self.monotonicity_width},
                "value": float(self.monotonicity.compute().item()),
            }]
            metrics_list.append({
                "name": "Monotonicity",
                "variants": mono_variants,
            })

        return {"metrics": metrics_list}

    def _save_statistic_plots(self) -> None:
        """Save top-k bar chart figures for metrics with largest k."""
        save_dir = self.save_path.parent / "statistics"
        save_dir.mkdir(parents=True, exist_ok=True)
        K = max(self.token_topk) if self.token_topk else 0
        KC = max(self.pair_topk) if self.pair_topk else 0

        for tol in self.ber_tols:
            for mode in ("onset", "offset"):
                if K:
                    self._plot_boundary_topk(
                        self.metrics[f"ber/{mode}/{tol}/{K}"],
                        key=f"ber/{mode}/{tol}/{K}", save_dir=save_dir,
                    )
        if K:
            for mode in ("onset", "offset"):
                self._plot_boundary_topk(
                    self.metrics[f"b_mae/{mode}/{K}"],
                    key=f"b_mae/{mode}/{K}", save_dir=save_dir,
                )
            self._plot_overlap_topk(
                self.metrics[f"overlap/{K}"],
                key=f"overlap/{K}", save_dir=save_dir,
            )
        if KC:
            self._plot_conjunction_topk(
                self.metrics[f"conj_mae/{KC}"],
                key=f"conj_mae/{KC}", save_dir=save_dir,
            )

    def _plot_boundary_topk(self, metric, key, save_dir) -> None:
        top = metric.compute_top_k()
        if not top:
            return
        labels = [
            self._vocab.decode(tid, stringfy=True) or str(tid)
            for tid in top
        ]
        values = [v.item() for v in top.values()]
        safe_key = key.replace("/", "_")
        fig = topk_bar_figure(labels, values, key)
        fig.savefig(save_dir / f"{safe_key}.jpg")
        plt.close(fig)

    def _plot_overlap_topk(self, metric, key, save_dir) -> None:
        top = metric.compute_top_k()
        if not top:
            return
        safe_key = key.replace("/", "_")
        for sub_name, sub_data in top.items():
            labels = [
                self._vocab.decode(tid, stringfy=True) or str(tid)
                for tid in sub_data
            ]
            values = [v.item() for v in sub_data.values()]
            safe_sub = f"{safe_key}_{sub_name.replace('/', '_')}"
            fig = topk_bar_figure(labels, values, sub_name, reverse=False)
            fig.savefig(save_dir / f"{safe_sub}.jpg")
            plt.close(fig)

    def _plot_conjunction_topk(self, metric, key, save_dir) -> None:
        top = metric.compute_top_k()
        if not top:
            return
        labels = [
            f"{self._vocab.decode(i, stringfy=True) or str(i)} -> "
            f"{self._vocab.decode(j, stringfy=True) or str(j)}"
            for (i, j) in top
        ]
        values = [v.item() for v in top.values()]
        safe_key = key.replace("/", "_")
        fig = topk_bar_figure(labels, values, key)
        fig.savefig(save_dir / f"{safe_key}.jpg")
        plt.close(fig)
