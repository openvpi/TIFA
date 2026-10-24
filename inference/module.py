import json
import pathlib

import lightning.pytorch as pl
import matplotlib.pyplot as plt
import torch
from torchmetrics import MeanMetric

from inference.backend import InferenceBackend, SpectrogramContext
from lib import logging
from lib.path_traversal import materialize_paths
from modules.losses import FrameIdentityLoss
from modules.metrics.voice_event import VoiceEventPRF


class ForcedAlignmentInferenceModule(pl.LightningModule):
    """Forced alignment inference. Works with any InferenceBackend."""

    def __init__(
            self, backend: InferenceBackend,
            *,
            score_unit: str = "levenshtein",
            skip_penalty: float = 0.5,
    ):
        super().__init__()
        self.backend = backend
        self.score_unit = score_unit
        self.skip_penalty = skip_penalty

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

            phonemes = [
                label
                for candidates, choice in zip(batch["lexicon"][i], scored.choices[i, :W_i].tolist())
                if choice > 0
                for label in candidates[choice - 1]["phonemes"]
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
                "phonemes": phonemes,
                "texts": batch["texts"][i],
                "lexicon": batch["lexicon"][i],
            })

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
    def __init__(self, model, events: list[str], timestep, output_dir, identifiers, metric_config, plot=False):
        super().__init__()
        self.model = model
        self.events = events
        self.output_dir = pathlib.Path(output_dir)
        self.identifiers = identifiers
        self.metric_config = metric_config
        self.plot = plot
        self.metrics = VoiceEventPRF(
            events, collar=metric_config.event_collar_ms / (1000 * timestep),
            require_offset=metric_config.require_offset,
        )
        self.loss = FrameIdentityLoss()
        self.mean_loss = MeanMetric()

    def forward(self, batch):
        features = batch["spectrogram"]
        mask = torch.arange(features.shape[1], device=features.device)[None, :] < batch["T"][:, None]
        return self.model(features, mask), mask

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
