import json
import pathlib

import matplotlib.pyplot as plt
import torch
from torch import nn

from lib.config.io import load_raw_config
from lib.config.schema import BinarizerFeaturesConfig, RootConfig
from modules.losses import FrameIdentityLoss
from modules.metrics.voice_event import VoiceEventPRF
from modules.event_detection import VoiceEventDetectionModel
from training.pl_module_base import BaseLightningModule, LossValue
from training.data import VoiceEventDataset


class VoiceEventDetectorModule(BaseLightningModule):
    @classmethod
    def resolve_data_dirs(cls, config: RootConfig) -> tuple[pathlib.Path, None]:
        return config.binarizer.voice_event_data_dir_resolved, None

    def load_vocabulary(self) -> None:
        with open(self.data_dir / "vocabulary.json", "r", encoding="utf8") as f:
            self.events = json.load(f)["events"]
        if len(self.events) + 1 != self.model_config.num_classes:
            raise ValueError("Event vocabulary size does not match model.num_classes. Re-binarize the VED dataset.")

    def build_model(self) -> nn.Module:
        return VoiceEventDetectionModel(self.model_config)

    def register_losses_and_metrics(self):
        self.register_loss("voice_event_loss", FrameIdentityLoss(), weight=self.training_config.loss.voice_event.weight)
        features = BinarizerFeaturesConfig.model_validate(load_raw_config(self.data_dir / "feature.yaml"))
        self.timestep = features.timestep
        cfg = self.training_config.validation.voice_event
        for postfix in (["", "_dirty"] if self.use_parallel_dirty_metrics else [""]):
            self.register_metric(f"VoiceEvent{postfix}", VoiceEventPRF(
                self.events,
                collar=cfg.event_collar_ms / (1000 * self.timestep),
                require_offset=cfg.require_offset,
                postfix=postfix,
            ))

    def build_train_dataset(self):
        dl_cfg = self.training_config.dataloader
        return VoiceEventDataset(
            self.data_dir, "train",
            augmentation_config=self.training_config.augmentation,
            augmentation_seed=self.training_config.trainer.seed,
            max_concat_size=dl_cfg.max_concat_size,
            max_concat_frames=dl_cfg.max_concat_frames,
        )

    def build_valid_dataset(self):
        return VoiceEventDataset(
            self.data_dir, "valid",
            augmentation_config=(self.training_config.augmentation.keep_destructive()
                                 if self.use_parallel_dirty_metrics else None),
            augmentation_deterministic=True,
            augmentation_return_dirty=self.use_parallel_dirty_metrics,
        )

    def _forward(self, sample, spectrogram_key="spectrogram"):
        spectrogram = sample[spectrogram_key]
        mask = torch.arange(spectrogram.shape[1], device=spectrogram.device)[None, :] < sample["T"][:, None]
        return self.model(spectrogram, mask), mask

    def _update_metrics(self, sample, logits, mask, postfix=""):
        self.metrics[f"VoiceEvent{postfix}"].update(
            logits.argmax(dim=-1), sample["frame_targets"], mask,
            sample["event_labels"], sample["event_spans"],
        )

    def forward_model(self, sample, infer: bool, batch_idx=None):
        logits, mask = self._forward(sample)
        if infer:
            self._update_metrics(sample, logits, mask)
            return {"probabilities": logits.float().softmax(dim=-1)}
        group_count = 0
        if batch_idx is not None:
            self.train_sampler.form_batches()
            group_count = sum(self.train_dataset.num_frames(index)
                              for batch in self.get_accumulation_group(batch_idx) for index in batch)
        return {"voice_event_loss": LossValue(
            self.losses["voice_event_loss"](logits, sample["frame_targets"], mask),
            int(mask.sum()), group_count,
        )}

    def validation_step(self, sample, batch_index):
        if sample["size"] == 0:
            return
        with torch.autocast(self.device.type, enabled=self.training_config.validation.allow_amp):
            logits, mask = self._forward(sample)
            loss = self.losses["voice_event_loss"](logits, sample["frame_targets"], mask)
            count = mask.sum()
            for metric in self.val_losses.values():
                metric.update(loss, weight=count)
            self._update_metrics(sample, logits, mask)
            if self.use_parallel_dirty_metrics:
                dirty_logits, _ = self._forward(sample, "spectrogram_dirty")
                self._update_metrics(sample, dirty_logits, mask, "_dirty")
            if self.global_rank == 0 and any(index in self.plot_indices for index in sample["indices"].tolist()):
                self.plot_validation_results(sample, {"probabilities": logits.float().softmax(dim=-1)})

    def plot_validation_results(self, sample, outputs):
        names = self.events
        for row, index in enumerate(sample["indices"].tolist()):
            if index not in self.plot_indices:
                continue
            length = int(sample["T"][row])
            probability = outputs["probabilities"][row, :length].detach().cpu().numpy().T
            target = sample["frame_targets"][row, :length].detach().cpu().numpy()
            fig, axes = plt.subplots(2, 1, figsize=(12, 4), sharex=True, gridspec_kw={"height_ratios": [4, 1]})
            axes[0].imshow(probability, origin="lower", aspect="auto", vmin=0, vmax=1)
            axes[0].set_yticks(range(len(names) + 1), [0, *names])
            axes[0].set_title(str(self.valid_dataset.get_metadata("item_paths", index)))
            axes[1].imshow(target[None, :], aspect="auto", vmin=0, vmax=len(names), interpolation="nearest")
            axes[1].set_ylabel("Target")
            axes[1].set_yticks([])
            axes[1].set_xlabel("Frame")
            fig.tight_layout()
            self.logger.experiment.add_figure(f"voice_event/{index}", fig, global_step=self.global_step)
            plt.close(fig)
