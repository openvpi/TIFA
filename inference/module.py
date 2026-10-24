import json
import pathlib

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
from torchmetrics import MeanMetric

from inference.backend import InferenceBackend, SpectrogramContext
from lib import logging
from lib.config.schema import BinarizerFeaturesConfig, VoiceEventValidationConfig
from lib.feature.mel import StretchableMelSpectrogram
from lib.path_traversal import materialize_paths
from modules.losses import FrameIdentityLoss
from modules.metrics.voice_event import VoiceEventPRF, frame_labels_to_events


def _insert_voice_events(result: dict, events, event_tokens: dict[str, int], timestep: float) -> bool:
    """Discard within-group events and split covering events at whole groups."""
    selected = [(name, start, end) for name, start, end in events if name in event_tokens]
    if not selected:
        return False

    tokens = result["tokens"].tolist()
    words = result["words"].tolist()
    groups = result["groups"].tolist()
    # Convert in Python to avoid float32 drift at matching event/group boundaries.
    spans = [(start * timestep, end * timestep) for start, end in result["spans"].tolist()]
    duration = result["spectrogram"].shape[0] * timestep
    size = len(tokens)
    slots = [0] + [i for i in range(1, size) if groups[i] != groups[i - 1]]
    if size:
        slots.append(size)
    group_spans = [
        (spans[left][0], spans[right - 1][1])
        for left, right in zip(slots, slots[1:])
        if spans[left][0] < spans[right - 1][1]
    ]
    gaps = [(spans[i - 1][1] if i else 0., spans[i][0] if i < size else duration) for i in slots]
    insertions = {i: [] for i in slots}
    for name, start, end in selected:
        if end <= start or any(left <= start and end <= right for left, right in group_spans):
            continue

        fragments = []
        cursor = start
        for left, right in group_spans:
            if start <= left and right <= end:
                if cursor < left:
                    fragments.append((cursor, left))
                cursor = right
        if cursor < end:
            fragments.append((cursor, end))

        for fragment_start, fragment_end in fragments:
            if any(left <= fragment_start and fragment_end <= right for left, right in group_spans):
                continue
            for slot, (left, right) in zip(slots, gaps):
                overlaps = max(fragment_start, left) < min(fragment_end, right)
                crosses = left == right and fragment_start < left < fragment_end
                if overlaps or crosses:
                    insertions[slot].append(name)
                    break

    if not any(insertions.values()):
        return False

    new_tokens, new_words, new_groups = [], [], []
    phonemes, texts, scripts = [], [], []
    word_ids, group_ids = {}, {}
    for i in range(size + 1):
        for name in insertions.get(i, ()):
            if 0 < i < size and words[i - 1] == words[i]:
                word_id = word_ids[words[i]]
            else:
                texts.append(name)
                word_id = len(texts)
            scripts.append(name)
            new_tokens.append(event_tokens[name])
            new_words.append(word_id)
            new_groups.append(len(scripts))
            phonemes.append(name)
        if i == size:
            break
        if words[i] not in word_ids:
            texts.append(result["texts"][words[i] - 1])
            word_ids[words[i]] = len(texts)
        if groups[i] not in group_ids:
            scripts.append(result["group_scripts"][groups[i] - 1])
            group_ids[groups[i]] = len(scripts)
        new_tokens.append(tokens[i])
        new_words.append(word_ids[words[i]])
        new_groups.append(group_ids[groups[i]])
        phonemes.append(result["phonemes"][i])

    result.update(
        tokens=result["tokens"].new_tensor(new_tokens),
        words=result["words"].new_tensor(new_words),
        groups=result["groups"].new_tensor(new_groups),
        phonemes=phonemes, texts=texts, group_scripts=scripts,
    )
    return True


class ForcedAlignmentInferenceModule(pl.LightningModule):
    """Forced alignment inference. Works with any InferenceBackend."""

    def __init__(
            self, backend: InferenceBackend,
            *,
            score_unit: str = "levenshtein",
            skip_penalty: float = 0.5,
            ved: "VoiceEventInferenceModule | None" = None,
            ved_event_tokens: dict[str, int] | None = None,
    ):
        super().__init__()
        self.backend = backend
        self.score_unit = score_unit
        self.skip_penalty = skip_penalty
        self.ved = ved
        self.ved_event_tokens = ved_event_tokens or {}

    def predict_step(self, batch, batch_idx):
        for msg in batch.get("warning", []):
            logging.warning(msg, callback=self.trainer.progress_bar_callback.print)

        if "waveform" not in batch:
            return []

        waveform = batch["waveform"]  # [B, L]
        duration = batch["duration"]  # [B] seconds
        paths = batch["paths"]

        # ---- Spectrogram ----
        spec = self.backend.spectrogram(waveform, duration)

        # ---- Score ----
        scored = self.backend.score(
            spec, paths=paths, words=batch["words"], candidates=batch["candidates"], unit=self.score_unit,
        )
        tokens, words, groups = materialize_paths(paths, batch["words"], batch["groups"], scored.choices)

        # ---- Align ----
        result = self.backend.align(
            spec, tokens=tokens, groups=groups, unit="frame", skip_penalty=self.skip_penalty,
        )

        # ---- Assemble per-item results ----
        results = []
        for i, identifier in enumerate(batch["identifier"]):
            N_i = int((tokens[i] != 0).sum().item())
            T_i = int(spec.mask[i].sum().item())
            W_i = len(batch["lexicon"][i])

            spec_i = spec.features[i, :T_i]
            tokens_i = tokens[i, :N_i]
            groups_i = groups[i, :N_i]
            spans_i = result.spans[i, :N_i]
            sim_i = result.similarity[i, :T_i, :N_i]
            agreement_i = result.agreement[i].item()

            selected = [
                candidates[choice - 1]
                for candidates, choice in zip(batch["lexicon"][i], scored.choices[i, :W_i].tolist())
                if choice > 0
            ]
            results.append({
                "identifier": identifier,
                "spectrogram": spec_i,
                "tokens": tokens_i,
                "groups": groups_i,
                "words": words[i, :N_i],
                "choices": scored.choices[i, :W_i],
                "scores": scored.scores[i, :W_i] if scored.scores is not None else None,
                "spans": spans_i,
                "similarity": sim_i,
                "agreement": agreement_i,
                "phonemes": [label for candidate in selected for label in candidate["phonemes"]],
                "group_scripts": [script for candidate in selected for script in candidate["scripts"]],
                "texts": batch["texts"][i],
                "score_texts": batch["texts"][i],
                "lexicon": batch["lexicon"][i],
            })

        if self.ved is not None:
            detected = self.ved.detect(batch["ved_waveform"], duration)
            changed = [
                i for i, (item, events) in enumerate(zip(results, detected))
                if _insert_voice_events(item, events, self.ved_event_tokens, self.backend.timestep)
            ]
            if changed:
                tokens = torch.nn.utils.rnn.pad_sequence(
                    [results[i]["tokens"] for i in changed], batch_first=True,
                )
                groups = torch.nn.utils.rnn.pad_sequence(
                    [results[i]["groups"] for i in changed], batch_first=True,
                )
                aligned = self.backend.align(
                    SpectrogramContext(spec.features[changed], spec.mask[changed]),
                    tokens=tokens, groups=groups, unit="frame", skip_penalty=self.skip_penalty,
                )
                for row, i in enumerate(changed):
                    item = results[i]
                    N_i, T_i = len(item["tokens"]), len(item["spectrogram"])
                    item.update(
                        spans=aligned.spans[row, :N_i],
                        similarity=aligned.similarity[row, :T_i, :N_i],
                        agreement=aligned.agreement[row].item(),
                    )

        return results

    def test_step(self, batch, batch_idx) -> dict:
        """Online evaluation on PhonemeTimingDataset batches.

        Returns predicted spans in frames so they share the same unit
        as ground truth (batch["spans"]).  The callback then applies a
        single unit_size_ms to convert both to ms.
        """
        spectrogram = batch["spectrogram"]  # [B, T, C]
        tokens = batch["tokens"]  # [B, N]
        T = spectrogram.shape[1]
        device = spectrogram.device

        mask = torch.arange(T, device=device).unsqueeze(0) < batch["T"].unsqueeze(1)
        spec = SpectrogramContext(features=spectrogram, mask=mask)
        result = self.backend.align(spec, tokens=tokens, unit="frame", skip_penalty=self.skip_penalty)
        return {"spans": result.spans, "similarity": result.similarity}


class VoiceEventInferenceModule(pl.LightningModule):
    def __init__(
            self, model, events: list[str], features: BinarizerFeaturesConfig,
            output_dir=None, identifiers=None, metric_config=None, plot=False,
    ):
        super().__init__()
        self.model = model
        self.events = events
        self.timestep = features.timestep
        self.spec_fn = StretchableMelSpectrogram(
            sample_rate=features.audio_sample_rate,
            n_mels=features.spectrogram.num_bins,
            n_fft=features.fft_size,
            win_length=features.win_size,
            hop_length=features.hop_size,
            fmin=features.spectrogram.fmin,
            fmax=features.spectrogram.fmax,
        )
        self.output_dir = pathlib.Path(output_dir) if output_dir is not None else None
        self.identifiers = identifiers
        self.metric_config = metric_config or VoiceEventValidationConfig()
        self.plot = plot
        self.metrics = VoiceEventPRF(
            events, collar=self.metric_config.event_collar_ms / (1000 * self.timestep),
            require_offset=self.metric_config.require_offset,
        )
        self.loss = FrameIdentityLoss()
        self.mean_loss = MeanMetric()

    def forward(self, batch):
        features = batch["spectrogram"]
        mask = torch.arange(features.shape[1], device=features.device)[None, :] < batch["T"][:, None]
        return self.model(features, mask), mask

    def detect(self, waveform, duration) -> list[list[tuple[str, float, float]]]:
        """Return event names and spans in seconds for whole-utterance audio."""
        features = self.spec_fn(waveform).permute(0, 2, 1)
        lengths = duration.div(self.timestep).round().long()
        logits, mask = self({"spectrogram": features, "T": lengths})
        labels = logits.argmax(-1).masked_fill(~mask, 0).cpu().tolist()
        return [
            [(self.events[label - 1], start * self.timestep, end * self.timestep)
             for label, start, end in frame_labels_to_events(frames)]
            for frames in labels
        ]

    def on_test_epoch_start(self):
        self.metrics.reset()
        self.mean_loss.reset()

    def test_step(self, batch, batch_idx):
        logits, mask = self(batch)
        self.metrics.update(logits.argmax(-1), batch["frame_targets"], mask,
                            batch["event_labels"], batch["event_spans"])
        self.mean_loss.update(self.loss(logits, batch["frame_targets"], mask), weight=mask.sum())
        if self.plot:
            self._save_plots(batch, logits)

    def on_test_epoch_end(self):
        values = {key: float(value) for key, value in self.metrics.compute().items()}
        values["voice_event_loss"] = float(self.mean_loss.compute())
        self.log_dict(values)
        summary = {
            "classes": self.events,
            "event_collar_ms": self.metric_config.event_collar_ms,
            "require_offset": self.metric_config.require_offset,
            "metrics": values,
            "counts": {
                name: {"frame": frames, "event": events}
                for name, frames, events in zip(self.events,
                                               self.metrics.frame_counts.cpu().tolist(),
                                               self.metrics.event_counts.cpu().tolist())
            },
            "count_order": ["tp", "fp", "fn"],
        }
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")

    def _save_plots(self, batch, logits):
        probabilities = logits.float().softmax(-1).detach().cpu().numpy()
        for row, index in enumerate(batch["indices"].tolist()):
            length = int(batch["T"][row])
            fig, ax = plt.subplots(figsize=(12, 3))
            ax.imshow(probabilities[row, :length].T, origin="lower", aspect="auto", vmin=0, vmax=1)
            ax.set_yticks(range(len(self.events) + 1), [0, *self.events])
            ax.set_xlabel("Frame")
            ax.set_title(self.identifiers[index])
            path = self.output_dir / "plots" / f"{index}.jpg"
            path.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(path, bbox_inches="tight")
            plt.close(fig)


class OfflineEvaluationModule(pl.LightningModule):
    """Passes paired TextGrid spans to callbacks.

    Batch (from PairedDataset) contains both spans_pred and spans_gt.
    Returns batch["spans_pred"] so the callback sees outputs["spans"].
    """

    def test_step(self, batch, batch_idx) -> dict:
        return {"spans": batch["spans_pred"]}
