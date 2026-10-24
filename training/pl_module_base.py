import abc
import pathlib
from typing import Any, NamedTuple
from fnmatch import fnmatch

import lightning.pytorch
import matplotlib
import numpy
import torch
from lightning_utilities.core.rank_zero import rank_zero_info
from torch import nn
from torchmetrics import MeanMetric, Metric
import tqdm

from lib import logging
from lib.config.schema import ModelConfig, RootConfig, TrainingConfig
from lib.reflection import build_lr_scheduler_from_config, build_optimizer_from_config
from lib.vocabulary import Vocabulary
from .data import BaseDataset, DynamicBatchSampler
from .weight_averaging import ExponentialMovingAverage

__all__ = [
    "BaseLightningModule",
    "LossValue",
]
matplotlib.use("Agg")  # fix Tcl_AsyncDelete: async handler deleted by the wrong thread


class LossValue(NamedTuple):
    mean: torch.Tensor  # per-valid-element mean (interpretable, for logging)
    batch_count: float  # weighted valid elements in THIS micro-batch
    group_count: float  # weighted valid elements across the accumulation group


class BaseLightningModule(lightning.pytorch.LightningModule, abc.ABC):

    def __init__(
            self,
            data_dir: pathlib.Path,
            model_config: ModelConfig,
            training_config: TrainingConfig,
            load_pretrained: bool = False,
            aux_data_dir: pathlib.Path | None = None,
    ):
        super().__init__()

        self.data_dir = data_dir
        self.aux_data_dir = aux_data_dir
        self.model_config = model_config
        self.training_config = training_config
        self.load_vocabulary()

        self.model: nn.Module = self.build_model()
        self.losses: dict[str, nn.Module] = nn.ModuleDict()
        self.metrics: dict[str, Metric] = nn.ModuleDict()
        self.loss_weights: dict[str, float] = {}
        self.val_losses: dict[str, Metric] = {  # use built-in dict to not be printed in the model summary
            "total_loss": MeanMetric()
        }
        self.plot_indices: set[int] = set()  # validation sample indices to plot after each epoch
        self.use_parallel_dirty_metrics = (
                self.training_config.augmentation.has_destructive_augmentations
                and self.training_config.validation.parallel_dirty_metrics
        )
        self.register_losses_and_metrics()
        if len(self.losses) == 0:
            raise ValueError("No losses defined.")

        self.freeze_parameters()  # caution: this can break when resuming training

        self.use_ema = self.training_config.weight_averaging.ema_enabled
        if self.use_ema:
            self.ema: ExponentialMovingAverage = self.build_ema()
        if load_pretrained and self.training_config.finetuning.pretraining_enabled:
            self.load_from_pretrained_model(training_config.finetuning.pretraining_from)

        self.train_dataset: BaseDataset = None
        self.valid_dataset: BaseDataset = None
        self.train_sampler: DynamicBatchSampler = None
        self.aux_dataset: BaseDataset = None
        self.aux_sampler: DynamicBatchSampler = None

        self.logger_step = -1  # when accumulate_grad_batches > 1, this helps to avoid redundant logging

        self.post_init()

    def load_vocabulary(self) -> None:
        """Load task vocabulary before building the model."""
        self.vocab = Vocabulary.from_file(self.data_dir / "vocabulary.json")
        if self.vocab.vocab_size > self.model_config.max_vocab_size:
            raise ValueError(
                f"Vocabulary size {self.vocab.vocab_size} exceeds "
                f"max_vocab_size {self.model_config.max_vocab_size}"
            )

    @classmethod
    @abc.abstractmethod
    def resolve_data_dirs(cls, config: RootConfig) -> tuple[pathlib.Path, pathlib.Path | None]:
        """Return (data_dir, aux_data_dir) from root config."""
        pass

    @abc.abstractmethod
    def build_model(self) -> nn.Module:
        """Build and return the model."""
        pass

    @abc.abstractmethod
    def register_losses_and_metrics(self) -> None:
        """
        Register the losses and metrics.
        Use `self.register_loss(name, loss)` to register a loss,
        and `self.register_metric(name, metric)` to register a metric.
        Registered losses and metrics can be accessed via `self.losses` and `self.metrics`.
        """
        pass

    def post_init(self) -> None:
        """This method is called after the model is initialized, useful for custom initialization logic."""
        pass

    def get_accumulation_group(self, batch_idx: int) -> list[list[int]]:
        """Return the sample-indices lists for all micro-batches in this batch's accumulation group."""
        A = self.training_config.trainer.accumulate_grad_batches
        group_start = (batch_idx // A) * A
        return self.train_sampler.batches[group_start:group_start + A]

    @abc.abstractmethod
    def forward_model(
            self, sample: dict[str, torch.Tensor], infer: bool, batch_idx: int | None = None
    ) -> dict[str, LossValue]:
        """
        Forward pass of the model.
        :param sample: the training or validation batch.
        :param infer: whether in inference mode.
        :param batch_idx: the batch index (provided during training for accumulation-group lookups).
        :return: if `infer` is True, update all registered metrics and return the model outputs;
            otherwise, return a dictionary mapping loss names to LossValue named tuples.
        """
        pass

    @abc.abstractmethod
    def plot_validation_results(self, sample: dict[str, torch.Tensor], outputs: dict[str, torch.Tensor]) -> None:
        """
        Plot the validation results on external logger like TensorBoard.
        :param sample: the validation batch.
        :param outputs: the model outputs returned by `self.forward_model(sample, infer=True)` directly.
        """
        pass

    def plot_validation_metrics(self) -> None:
        """Generate aggregate metric plots (bar charts, etc.).

        Called after all metrics have been computed and synced across ranks,
        so ``compute_top_k()`` can read globally-aggregated state directly.
        Override in subclasses that register top-k metrics.
        """
        pass

    def freeze_parameters(self):
        if not self.training_config.finetuning.freezing_enabled:
            return
        param_dict = dict(self.named_parameters())
        size = len(param_dict)
        param_dict = _apply_include_exclude(
            param_dict,
            includes=self.training_config.finetuning.freezing_include_params,
            excludes=self.training_config.finetuning.freezing_exclude_params
        )
        if len(param_dict) == size:
            raise ValueError(
                "Freezing all parameters is not allowed."
            )
        for param in param_dict.values():
            param.requires_grad = False
        logging.info(f"Freezing {len(param_dict)} parameter(s).", callback=rank_zero_info)

    def build_ema(self) -> ExponentialMovingAverage:
        parameters = dict(self.named_parameters())
        parameters = _apply_include_exclude(
            parameters,
            includes=self.training_config.weight_averaging.ema_include_params,
            excludes=self.training_config.weight_averaging.ema_exclude_params
        )
        ema = ExponentialMovingAverage(
            parameters=parameters,
            decay=self.training_config.weight_averaging.ema_decay
        )
        logging.info(f"EMA: {ema.size()} parameter(s) registered.", callback=rank_zero_info)
        return ema

    def load_from_pretrained_model(self, pretrained_model_path: pathlib.Path) -> None:
        ckpt = torch.load(
            pretrained_model_path, map_location=self.device, weights_only=True
        )
        source_state_dict = ckpt["state_dict"]
        source_state_dict = _apply_include_exclude(
            source_state_dict,
            includes=self.training_config.finetuning.pretraining_include_params,
            excludes=self.training_config.finetuning.pretraining_exclude_params
        )
        target_state_dict = self.state_dict()
        for name in list(source_state_dict.keys()):
            if name not in target_state_dict:
                del source_state_dict[name]
        _check_shape_consistency(
            source_state_dict, target_state_dict,
            error_message=f"Pretrained model '{pretrained_model_path}' has mismatched parameter(s)"
        )
        self.load_state_dict(source_state_dict, strict=False)
        logging.info(
            f"Loaded {len(source_state_dict)} parameter(s) from '{pretrained_model_path}'",
            callback=rank_zero_info
        )
        if self.use_ema:
            self.ema.register()  # copy to shadow again after updating referenced parameters
            source_ema_state_dict = ckpt.get("ema_state_dict", {})
            target_ema_state_dict = self.ema.state_dict()
            for name in list(source_ema_state_dict.keys()):
                if name not in source_state_dict or name not in target_ema_state_dict:
                    del source_ema_state_dict[name]
            _check_shape_consistency(
                source_ema_state_dict, target_ema_state_dict,
                error_message=f"Pretrained model '{pretrained_model_path}' has mismatched EMA parameter(s)"
            )
            self.ema.load_state_dict(source_ema_state_dict, strict=False)
            logging.info(
                f"Loaded {len(source_ema_state_dict)} EMA parameter(s) from '{pretrained_model_path}'",
                callback=rank_zero_info
            )

    def register_loss(self, name: str, loss: nn.Module, weight: float = 1.0, validation: bool = True) -> None:
        """
        Register a loss module that can be accessed via `self.losses`.
        The *weight* is applied in `training_step`; `forward_model` should
        return raw (unweighted) loss means.
        """
        if name in self.losses:
            raise ValueError(f"Loss '{name}' already registered.")
        if name in self.metrics:
            raise ValueError(f"Loss name '{name}' is already used by a metric.")
        self.losses[name] = loss
        self.loss_weights[name] = weight
        if validation:
            self.val_losses[name] = MeanMetric()  # for validation logging

    def register_metric(self, name: str, metric: Metric) -> None:
        """
        Register a metric that can be accessed via `self.metrics`.
        """
        if name in self.metrics:
            raise ValueError(f"Metric '{name}' already registered.")
        if name in self.losses:
            raise ValueError(f"Metric name {name} is already used by a loss.")
        self.metrics[name] = metric

    @abc.abstractmethod
    def build_train_dataset(self) -> BaseDataset:
        """Build the training dataset."""
        pass

    @abc.abstractmethod
    def build_valid_dataset(self) -> BaseDataset:
        """Build the validation dataset."""
        pass

    def build_aux_dataset(self) -> BaseDataset | None:
        """Build the aux dataset. Defaults to None; override in subclasses that need aux data."""
        return None

    def setup(self, stage: str) -> None:
        if stage != "fit":
            raise ValueError("This module only supports the 'fit' stage.")
        self.train_dataset = self.build_train_dataset()
        self.valid_dataset = self.build_valid_dataset()
        if self.aux_data_dir is not None:
            self.aux_dataset = self.build_aux_dataset()

    def on_fit_start(self) -> None:
        if self.use_ema:
            # DDP has synchronized the model, but does not manage EMA shadows.
            self.ema.synchronize()

    def train_dataloader(self):
        dataloader_config = self.training_config.dataloader
        self.train_sampler = DynamicBatchSampler(
            self.train_dataset,
            max_batch_size=dataloader_config.max_batch_size,
            max_batch_frames=dataloader_config.max_batch_frames,
            sort_by_len=True,
            frame_count_grid=dataloader_config.frame_count_grid,
            batch_count_multiple_of=self.training_config.trainer.accumulate_grad_batches,
            reassign_batches=True,
            shuffle_batches=True,
            seed=42,
        )
        return torch.utils.data.DataLoader(
            self.train_dataset,
            collate_fn=self.train_dataset.collate,
            batch_sampler=self.train_sampler,
            num_workers=dataloader_config.num_workers,
            prefetch_factor=dataloader_config.prefetch_factor if dataloader_config.num_workers > 0 else None,
            pin_memory=True,
            persistent_workers=dataloader_config.num_workers > 0
        )

    def val_dataloader(self):
        dataloader_config = self.training_config.dataloader
        return torch.utils.data.DataLoader(
            self.valid_dataset,
            collate_fn=self.valid_dataset.collate,
            batch_sampler=DynamicBatchSampler(
                self.valid_dataset,
                max_batch_size=dataloader_config.max_val_batch_size,
                max_batch_frames=dataloader_config.max_val_batch_frames,
                sort_by_len=False,
                reassign_batches=False,
                shuffle_batches=False,
            ),
            num_workers=dataloader_config.num_workers,
            prefetch_factor=dataloader_config.prefetch_factor if dataloader_config.num_workers > 0 else None,
            persistent_workers=dataloader_config.num_workers > 0
        )

    def configure_optimizers(self):
        optimizer = build_optimizer_from_config(self, self.training_config.optimizer)
        scheduler = build_lr_scheduler_from_config(optimizer, self.training_config.lr_scheduler)
        interval = self.training_config.lr_scheduler.unit
        if isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
            if interval != self.training_config.trainer.unit:
                # ReduceLROnPlateau requires the scheduler to synchronize with the validation period
                raise ValueError(
                    f"ReduceLROnPlateau scheduler requires training.lr_scheduler.unit and training.trainer.unit "
                    f"to be the same, got '{interval}' and '{self.training_config.trainer.unit}'."
                )
            # Call scheduler.step() after each validation
            frequency = self.training_config.trainer.val_every_n_units
            monitor = self.training_config.lr_scheduler.monitor
            if monitor not in self.val_losses and monitor not in self.metrics:
                raise ValueError(
                    f"Invalid monitor '{monitor}' for ReduceLROnPlateau scheduler. Should be one of "
                    f"losses {list(self.val_losses.keys())} or metrics {list(self.metrics.keys())}."
                )
        else:
            frequency = 1
            monitor = None
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": interval,
                "frequency": frequency,
                "monitor": monitor,
                "strict": False,  # in case the candidates are empty after resuming
            }
        }

    def on_train_epoch_start(self):
        if self.train_sampler is not None:
            self.train_sampler.set_epoch(self.current_epoch)

    def training_step(self, sample: dict[str, torch.Tensor], batch_index: int):
        loss_values = self.forward_model(sample, infer=False, batch_idx=batch_index)
        A = self.training_config.trainer.accumulate_grad_batches

        total_loss = 0.0
        for name, lv in loss_values.items():
            grad_weight = (A * lv.batch_count / lv.group_count) if lv.group_count > 0 else 1.0
            total_loss += lv.mean * self.loss_weights[name] * grad_weight

        unweighted_total = sum(lv.mean for lv in loss_values.values())
        if torch.isinf(unweighted_total) or torch.isnan(unweighted_total):
            detail = " ".join(
                f"{name}={lv.mean.item():.4f}(bc={lv.batch_count},gc={lv.group_count})"
                for name, lv in loss_values.items()
            )
            logging.warning(
                f"Non-finite total_loss at step {self.global_step}: {detail}",
                callback=self.trainer.progress_bar_callback.print,
            )
        log_outputs = {
            **{name: lv.mean for name, lv in loss_values.items()},
            "batch_size": sample["size"],
        }
        if "aux_size" in sample:
            log_outputs["aux_batch_size"] = sample["aux_size"]
        # logs to progress bar
        self.log("total_loss", unweighted_total, prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log("batch_size", sample["size"], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        self.log("lr", self.lr_schedulers().get_last_lr()[0], prog_bar=True, logger=False, on_step=True, on_epoch=False)
        # logs to tensorboard
        if (
                self.global_step > self.logger_step and
                self.global_step % self.training_config.trainer.log_every_n_steps == 0
        ):
            tb_log = {f"training/{k}": v for k, v in log_outputs.items()}
            tb_log["training/lr"] = self.lr_schedulers().get_last_lr()[0]
            tb_log["training/epoch"] = self.current_epoch
            self.logger.log_metrics(tb_log, step=self.global_step)
            self.logger_step = self.global_step
        return total_loss

    def optimizer_step(self, *args, **kwargs):
        super().optimizer_step(*args, **kwargs)
        if self.use_ema:
            self.ema.step()  # update EMA parameters after optimizer step

    def on_validation_epoch_start(self):
        for metric in self.val_losses.values():
            # self.val_losses is a built-in dict, so we need to move them to device manually
            metric.to(self.device)
            metric.reset()
        for metric in self.metrics.values():
            metric.reset()
        n = len(self.valid_dataset)
        k = min(n, self.training_config.validation.max_plots)
        self.plot_indices = set(numpy.linspace(0, n, k, endpoint=False, dtype=int).tolist())

    def validation_step(self, sample: dict[str, torch.Tensor], batch_index: int):
        if sample["size"] == 0:
            return
        save_obj = {}
        with torch.autocast(self.device.type, enabled=False):
            losses = self.forward_model(sample, infer=False)
            outputs = self.forward_model(sample, infer=True)
            if any(idx in self.plot_indices for idx in sample["indices"].tolist()):
                save_obj["sample"] = sample
                save_obj["outputs"] = outputs
                filename = f"validation_step{self.global_step}_rank{self.global_rank}_batch{batch_index}.pt"
                torch.save(
                    obj=save_obj,
                    f=pathlib.Path(self.logger.log_dir) / filename,
                )
            loss_means = {
                name: lv.mean for name, lv in losses.items()
            }
            loss_means = {
                "total_loss": sum(loss_means.values()),
                **loss_means,
            }
            for k, v in loss_means.items():
                self.val_losses[k].update(v, weight=sample["size"])

    def on_validation_epoch_end(self):
        loss_vals = {k: v.compute() for k, v in self.val_losses.items()}
        metric_vals = {}
        for k, v in self.metrics.items():
            m = v.compute()
            if isinstance(m, dict):
                metric_vals.update(m)
            else:
                metric_vals[k] = m
        self.log_dict(
            {**loss_vals, **metric_vals},
            on_epoch=True, prog_bar=False, logger=False, sync_dist=True
        )
        if self.global_rank != 0:
            return
        self.logger.log_metrics({f"validation/{k}": v for k, v in loss_vals.items()}, step=self.global_step)
        self.logger.log_metrics({f"validation/{k}": v for k, v in metric_vals.items()}, step=self.global_step)
        filelist = list(pathlib.Path(self.logger.log_dir).glob(f"validation_step{self.global_step}_rank*_batch*.pt"))
        with torch.autocast(self.device.type, enabled=self.training_config.validation.allow_amp):
            for file in tqdm.tqdm(filelist, desc="Plotting", leave=False):
                obj = torch.load(file, map_location=self.device, weights_only=True)
                sample = obj["sample"]
                outputs = obj["outputs"]
                self.plot_validation_results(sample, outputs)
                file.unlink()
        self.plot_validation_metrics()

    def on_save_checkpoint(self, checkpoint: dict[str, torch.Tensor]):
        if self.use_ema:
            checkpoint["ema_state_dict"] = self.ema.state_dict()

    def on_load_checkpoint(self, checkpoint: dict[str, torch.Tensor]):
        if self.use_ema:
            self.ema.load_state_dict(checkpoint.pop("ema_state_dict"), strict=True)


def _apply_include_exclude(
        dict_to_filter: dict[str, Any],
        includes: list[str] = None,
        excludes: list[str] = None,
) -> dict[str, Any]:
    result = {}
    for key, value in dict_to_filter.items():
        if includes and not any(fnmatch(key, pattern) for pattern in includes):
            continue
        if excludes and any(fnmatch(key, pattern) for pattern in excludes):
            continue
        result[key] = value
    return result


def _check_shape_consistency(
        source_state_dict: dict[str, torch.Tensor],
        target_state_dict: dict[str, torch.Tensor],
        error_message: str
):
    errors = []
    for name in list(source_state_dict.keys()):
        source_param = source_state_dict[name]
        target_param = target_state_dict[name]
        if source_param.shape != target_param.shape:
            errors.append((name, tuple(source_param.shape), tuple(target_param.shape)))
    if errors:
        raise RuntimeError(
            f"{error_message}:\n"
            + "\n".join(
                f"  {name}: source {source_shape}, target {target_shape}"
                for name, source_shape, target_shape in errors
            )
        )
