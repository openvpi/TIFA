import glob
import pathlib
from collections.abc import Callable
from typing import Annotated, Any, Literal, Union

import numpy as np
from g2pflow import G2PPipelineConfig
from pydantic import Field, PrivateAttr, field_validator

from .core import ConfigBaseModel
from .ops import (
    ConfigOperationBase, ConfigOperationContext,
    ref, this, ctx, if_, exists, coalesce
)


class ConfigurationScope:
    FA = 0x1
    FA_SSL = 0x2


class DynamicCheck:
    def __init__(self, expr: ConfigOperationBase, message=None):
        self.expr = expr
        self.message = message

    def run(self, context: ConfigOperationContext):
        if isinstance(self.expr, ConfigOperationBase):
            expr = self.expr.resolve(context)
        else:
            expr = self.expr
        if not expr:
            raise ValueError(
                f"Dynamic check failed.\n"
                f"{'.'.join(str(e) for e in context.current_path)}\n"
                f"  {self.message}"
            )


class RequiredOnGivenScope(DynamicCheck):
    def __init__(self, scope_mask: int):
        super().__init__(
            expr=if_(ctx("scope") & scope_mask, exists(this()), True),
            message="Field required."
        )


class VocabularyConfig(ConfigBaseModel):
    prebuilt_vocab_file: str | None = Field(default=None)
    global_symbols: list[str] = Field(default_factory=list)
    stop_symbols: list[str] = Field(default_factory=list)
    merged_groups: list[list[str]] = Field(default_factory=list)


class SpectrogramConfig(ConfigBaseModel):
    type: Literal["mel"] = Field("mel")
    num_bins: int = Field(128, gt=0)
    fmin: float = Field(0, ge=0)
    fmax: float = Field(8000, ge=0)


class F0Config(ConfigBaseModel):
    method: Literal["parselmouth"] = Field("parselmouth")
    f0_min: float = Field(65, gt=0)
    f0_max: float = Field(1600, gt=0, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=this() > ref("binarizer.features.f0.f0_min"),
            message="f0_max must be greater than f0_min."
        )
    })


class BinarizerFeaturesConfig(ConfigBaseModel):
    audio_sample_rate: int = Field(44100, gt=0)
    hop_size: int = Field(441, gt=0)
    fft_size: int = Field(2048, gt=0)
    win_size: int = Field(2048, gt=0)
    spectrogram: SpectrogramConfig = Field(...)
    f0: F0Config | None = Field(None, json_schema_extra={
        "scope": ConfigurationScope.FA_SSL,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.FA_SSL),
    })

    @property
    def timestep(self):
        return self.hop_size / self.audio_sample_rate

    # noinspection PyMethodParameters
    @field_validator("spectrogram")
    def check_kwargs(cls, v: SpectrogramConfig):
        if v.fmin >= v.fmax:
            raise ValueError("fmin must be less than fmax.")
        return v


class ValidationSplitConfig(ConfigBaseModel):
    subdir: str | None = Field(None)
    count: Union[
        Annotated[int, Field(lt=0)],
        Annotated[int, Field(gt=0)],
    ] = Field(
        100,
        json_schema_extra={
            "dynamic_check": DynamicCheck(
                expr=if_(this() < 0, exists(ref("binarizer.split.subdir")), True),
                message=(
                    "A negative count would select the entire dataset "
                    "for validation and leave the training set empty."
                ),
            )
        },
    )
    seed: int | None = Field(42)


class BinarizerConfig(ConfigBaseModel):
    phoneme_timing_data_dir: str = Field(None, json_schema_extra={
        "scope": ConfigurationScope.FA,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.FA),
    })
    text_only_data_dir: str | None = Field(None, json_schema_extra={
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.FA_SSL),
    })
    g2p: G2PPipelineConfig | None = Field(None)
    vocabulary: VocabularyConfig = Field(default_factory=VocabularyConfig)
    split: ValidationSplitConfig = Field(default_factory=ValidationSplitConfig)
    num_workers: int = Field(0, ge=0)
    features: BinarizerFeaturesConfig = Field(...)

    @property
    def phoneme_timing_data_dir_resolved(self) -> pathlib.Path:
        return pathlib.Path(self.phoneme_timing_data_dir).resolve()

    @property
    def text_only_data_dir_resolved(self) -> pathlib.Path | None:
        if self.text_only_data_dir is None:
            return None
        return pathlib.Path(self.text_only_data_dir).resolve()


class BackboneConfig(ConfigBaseModel):
    cls: str = Field(...)
    kwargs: dict[str, Any] = Field(...)


class ModelConfig(ConfigBaseModel):
    arch: str = Field(...)
    max_vocab_size: int = Field(256)
    in_dim: int = Field(None, json_schema_extra={
        "dynamic_expr": ref("binarizer.features.spectrogram.num_bins")
    })
    embedding_dim: int = Field(256, gt=0)
    out_dim: int = Field(256, gt=0, json_schema_extra={
        "scope": ConfigurationScope.FA,
    })
    backbone: BackboneConfig = Field(...)
    reconstructor: BackboneConfig | None = Field(None, json_schema_extra={
        "scope": ConfigurationScope.FA_SSL,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.FA_SSL),
    })


class PitchShiftingAugmentationConfig(ConfigBaseModel):
    enabled: bool = Field(False)
    prob: float = Field(0.5, gt=0.0, le=1.0)
    min_semitones: float = Field(-12.0)
    max_semitones: float = Field(12.0, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=this() > ref("training.augmentation.pitch_shifting.min_semitones"),
            message="max_semitones must be greater than min_semitones."
        )
    })


class TimeStretchingAugmentationConfig(ConfigBaseModel):
    enabled: bool = Field(False)
    prob: float = Field(0.5, gt=0.0, le=1.0)
    min_speed: float = Field(0.8, gt=0.0)
    max_speed: float = Field(1.25, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=this() > ref("training.augmentation.time_stretching.min_speed"),
            message="max_speed must be greater than min_speed."
        )
    })


class LoudnessScalingAugmentationConfig(ConfigBaseModel):
    enabled: bool = Field(False)
    prob: float = Field(0.5, gt=0.0, le=1.0)
    min_db: float = Field(-12.0)
    max_db: float = Field(12.0, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=this() > ref("training.augmentation.loudness_scaling.min_db"),
            message="max_db must be greater than min_db."
        )
    })


class SpectrogramMaskingAugmentationConfig(ConfigBaseModel):
    enabled: bool = Field(False)
    time_mask_prob: float = Field(0.15, gt=0.0, le=1.0)
    time_mask_max_width: int = Field(50, gt=0)
    freq_mask_prob: float = Field(0.15, gt=0.0, le=1.0)
    freq_mask_max_width: int = Field(20, gt=0, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=this() < ref("binarizer.features.spectrogram.num_bins"),
            message="freq_mask_max_width must be less than num_bins."
        )
    })
    intersect_prob: float = Field(0.5, gt=0.0, le=1.0)
    max_repeats: int = Field(1, ge=1)


class ColoredNoiseAugmentationConfig(ConfigBaseModel):
    enabled: bool = Field(False)
    prob: float = Field(0.25, gt=0.0, le=1.0)
    min_exponent: float = Field(0)
    max_exponent: float = Field(2.0, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=this() >= ref("training.augmentation.colored_noise.min_exponent"),
            message="max_exponent must be greater than or equal to min_exponent."
        )
    })


class NaturalNoiseAugmentationConfig(ConfigBaseModel):
    enabled: bool = Field(False)
    prob: float = Field(0.25, gt=0.0, le=1.0)
    max_repeats: int = Field(1, ge=1)
    noise_path_glob: str = Field("data/noise/**/*.wav")
    _noise_file_list: np.ndarray | None = PrivateAttr(default=None)

    @property
    def noise_file_list(self) -> np.ndarray:
        if self._noise_file_list is None:
            self._noise_file_list = np.array(glob.glob(self.noise_path_glob, recursive=True))
        return self._noise_file_list


class RIRReverbAugmentationConfig(ConfigBaseModel):
    enabled: bool = Field(False)
    prob: float = Field(0.25, gt=0.0, le=1.0)
    kernel_path_glob: str = Field("data/reverb/**/*.wav")
    _kernel_file_list: np.ndarray | None = PrivateAttr(default=None)

    @property
    def kernel_file_list(self) -> np.ndarray:
        if self._kernel_file_list is None:
            self._kernel_file_list = np.array(glob.glob(self.kernel_path_glob, recursive=True))
        return self._kernel_file_list


class SequenceEditAugmentationConfig(ConfigBaseModel):
    enabled: bool = False
    prob: float = Field(0.3, gt=0.0, le=1.0)
    p_sub: float = Field(0.08, gt=0.0, le=1.0)
    p_del: float = Field(0.08, gt=0.0, le=1.0)
    p_ins: float = Field(0.05, gt=0.0, le=1.0)


class TokenMaskingAugmentationConfig(ConfigBaseModel):
    enabled: bool = False
    prob: float = Field(0.2, gt=0.0, le=1.0)
    p_mask: float = Field(0.5, gt=0.0, le=1.0)
    p_insert: float = Field(0.05, ge=0.0, le=1.0)
    p_chain: float = Field(0.2, ge=0.0, lt=1.0)
    max_chain: int = Field(3, ge=1)


class AugmentationConfig(ConfigBaseModel):
    pitch_shifting: PitchShiftingAugmentationConfig = Field(...)
    time_stretching: TimeStretchingAugmentationConfig = Field(...)
    loudness_scaling: LoudnessScalingAugmentationConfig = Field(...)
    spectrogram_masking: SpectrogramMaskingAugmentationConfig = Field(...)
    colored_noise: ColoredNoiseAugmentationConfig = Field(...)
    natural_noise: NaturalNoiseAugmentationConfig = Field(...)
    rir_reverb: RIRReverbAugmentationConfig = Field(...)
    sequence_edit: SequenceEditAugmentationConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.FA,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.FA),
    })
    token_masking: TokenMaskingAugmentationConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.FA,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.FA),
    })

    @classmethod
    def destructive_augmentation_names(cls) -> set[str]:
        return {
            "spectrogram_masking",
            "colored_noise",
            "natural_noise",
            "rir_reverb",
        }

    @property
    def has_destructive_augmentations(self) -> bool:
        return (
                self.spectrogram_masking.enabled
                or self.colored_noise.enabled
                or self.natural_noise.enabled
                or self.rir_reverb.enabled
        )

    def _validate_names(self, names: tuple[str, ...]) -> set[str]:
        valid_fields = set(self.model_fields.keys())
        names_set = set(names)
        unknown = names_set - valid_fields
        if unknown:
            raise ValueError(
                f"Unknown augmentation types: {sorted(unknown)}. "
                f"Valid: {sorted(valid_fields)}"
            )
        return names_set

    def _clone_enabled(
            self, compute: Callable[[str, bool], bool]
    ) -> "AugmentationConfig":
        updates: dict[str, object] = {}
        for field_name in self.model_fields:
            if not hasattr(self, field_name) or (sub := getattr(self, field_name)) is None:
                continue
            new_enabled = compute(field_name, sub.enabled)
            if new_enabled != sub.enabled:
                updates[field_name] = sub.model_copy(
                    update={"enabled": new_enabled}
                )
        if not updates:
            return self
        return self.model_copy(update=updates)

    def drop(self, *names: str) -> "AugmentationConfig":
        """Return a copy with the named augmentations disabled."""
        names_set = self._validate_names(names)
        return self._clone_enabled(
            lambda n, e: False if n in names_set else e
        )

    def keep(self, *names: str) -> "AugmentationConfig":
        """Return a copy keeping only the named augmentations at their
        current ``enabled`` state.  All other types are disabled."""
        names_set = self._validate_names(names)
        return self._clone_enabled(
            lambda n, e: e and n in names_set
        )

    def keep_destructive(self) -> "AugmentationConfig":
        """Keep only the destructive augmentation types at their current
        ``enabled`` state.  Matches :attr:`has_destructive_augmentations`."""
        return self.keep(*self.destructive_augmentation_names())


class IterativeRankingConfig(ConfigBaseModel):
    enabled: bool = False
    rank_size: int = Field(4, ge=2)
    gamma: float = Field(0.9, gt=0.0, le=1.0)
    update_every_n_epochs: int = Field(1, ge=1)


class LossInstanceConfig(ConfigBaseModel):
    weight: float = Field(1.0)


class FrameAlignmentLossConfig(LossInstanceConfig):
    temperature: float = 0.1


class SpanContrastiveLossConfig(LossInstanceConfig):
    temperature: float = 0.1
    bidirectional: bool = True


class TokenIdentityLossConfig(LossInstanceConfig):
    pass


class FrameIdentityLossConfig(LossInstanceConfig):
    pass


class CTCLossConfig(LossInstanceConfig):
    pass


class HMMForwardLossConfig(LossInstanceConfig):
    mode: Literal["separate", "mean"] = Field("separate")


class SpectrogramReconstructionLossConfig(LossInstanceConfig):
    loss_type: Literal["L1", "L2"] = Field("L1")
    unmasked_weight: float = Field(0.01, gt=0.0)


class LossConfig(ConfigBaseModel):
    frame_alignment: FrameAlignmentLossConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.FA,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.FA),
    })
    span_contrastive: SpanContrastiveLossConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.FA,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.FA),
    })
    token_identity: TokenIdentityLossConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.FA,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.FA),
    })
    frame_identity: FrameIdentityLossConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.FA,
    })
    ctc: CTCLossConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.FA_SSL,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.FA_SSL),
    })
    hmm_forward: HMMForwardLossConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.FA_SSL,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.FA_SSL),
    })
    reconstruction: SpectrogramReconstructionLossConfig = Field(None, json_schema_extra={
        "scope": ConfigurationScope.FA_SSL,
        "dynamic_check": RequiredOnGivenScope(ConfigurationScope.FA_SSL),
    })


class DataLoaderConfig(ConfigBaseModel):
    max_concat_size: int | None = Field(None, ge=1)
    max_concat_frames: int | None = Field(None, ge=1)
    max_batch_size: int = Field(64, gt=0)
    max_batch_frames: int = Field(50000, gt=0)
    max_val_batch_size: int = Field(1, gt=0)
    max_val_batch_frames: int = Field(20000, gt=0)
    aux_ratio: float = Field(
        1.0,
        gt=0.0,
        description="Auxiliary-to-main batch budget ratio.",
    )
    aux_warmup_epochs: int = Field(0, ge=0)
    frame_count_grid: int = Field(6, ge=1)
    num_workers: int = Field(4, ge=0)
    prefetch_factor: int = Field(2, ge=0)


class OptimizerConfig(ConfigBaseModel):
    cls: str = Field(...)
    wraps: Literal["parameters", "module"] = Field("parameters")
    kwargs: dict[str, Any] = Field(...)


def _walk_lr_scheduler_configs(obj, fn):
    """Recursively apply *fn* to every LRSchedulerConfig-shaped entry in *obj*."""
    if isinstance(obj, LRSchedulerConfig):
        return fn(obj)
    if isinstance(obj, dict):
        if "cls" in obj:
            obj.setdefault("kwargs", {})
            return fn(LRSchedulerConfig.model_validate(obj))
        return {k: _walk_lr_scheduler_configs(v, fn) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_walk_lr_scheduler_configs(item, fn) for item in obj]
    return obj


class LRSchedulerConfig(ConfigBaseModel):
    cls: str = Field(...)
    kwargs: dict[str, Any] = Field(...)
    unit: Literal["step", "epoch"] = Field(...)
    monitor: str | None = Field(None)

    # noinspection PyMethodParameters
    @field_validator("kwargs")
    def check_kwargs(cls, v):
        return _walk_lr_scheduler_configs(v, lambda x: x)


class PeriodicCheckpointConfig(ConfigBaseModel):
    tag: str = Field(...)
    type: Literal["periodic"] = Field("periodic")
    unit: Literal["step", "epoch"] = Field(None, json_schema_extra={
        "dynamic_expr": coalesce(this(), ref("training.trainer.unit"))
    })
    since_m_units: int = Field(0, ge=0)
    every_n_units: int = Field(...)
    save_last_k: int = Field(1, ge=-1)
    weights_only: bool = Field(False)


class ExpressionCheckpointConfig(ConfigBaseModel):
    tag: str = Field(...)
    type: Literal["expression"] = Field("expression")
    expression: str = Field(...)
    save_top_k: int = Field(5, ge=-1)
    mode: Literal["max", "min"] = Field(...)
    weights_only: bool = Field(False)


ModelCheckpointConfig = Annotated[
    PeriodicCheckpointConfig | ExpressionCheckpointConfig,
    Field(discriminator="type")
]


class TrainerStrategyConfig(ConfigBaseModel):
    name: str = Field("auto")
    kwargs: dict[str, Any] = Field(...)


class TrainerConfig(ConfigBaseModel):
    seed: int | None = Field(None, ge=0)
    unit: Literal["step", "epoch"] = Field(...)
    min_steps: int = Field(0)
    max_steps: int = Field(160000)
    min_epochs: int = Field(0)
    max_epochs: int = Field(1000)
    val_every_n_units: int = Field(..., ge=1)
    log_every_n_steps: int = Field(100, ge=1)
    num_sanity_val_steps: int = Field(1)
    checkpoints: list[ModelCheckpointConfig] = Field(..., min_length=1)
    accelerator: str = Field("auto")
    devices: Union[Literal["auto"], int, list[int]] = Field("auto")
    num_nodes: Literal[1] = Field(1, ge=1)
    strategy: TrainerStrategyConfig = Field(...)
    precision: str = Field("16-mixed")
    accumulate_grad_batches: int = Field(1, ge=1)
    gradient_clip_val: float = Field(1.0, gt=0)

    # noinspection PyMethodParameters
    @field_validator("checkpoints")
    def check_checkpoints(cls, v):
        tags = set()
        for checkpoint in v:
            if checkpoint.tag in tags:
                raise ValueError(f"Duplicate checkpoint tag: '{checkpoint.tag}'.")
            tags.add(checkpoint.tag)
        if all(c.weights_only for c in v):
            raise ValueError("At least one checkpoint should set weights_only to False.")
        return v


class ValidationConfig(ConfigBaseModel):
    allow_amp: bool = Field(False)
    max_plots: int = Field(100, ge=0)
    skip_penalty: float = Field(0.5, ge=0.0)
    parallel_dirty_metrics: bool = Field(True)
    metrics_ber_tolerance: int = Field(5, ge=0)
    metrics_k_values: list[int] = Field([5, 20])
    metrics_conjunction_k_values: list[int] = Field([5, 20])
    metrics_determinacy_power: float = Field(2.0, ge=0)
    metrics_determinacy_width: int | None = Field(5, ge=0)
    metrics_monotonicity_power: float = Field(2.0, ge=0)
    metrics_monotonicity_width: int | None = Field(None, ge=0)


class FinetuningConfig(ConfigBaseModel):
    pretraining_enabled: bool = Field(False)
    pretraining_from: str | None = Field(None, json_schema_extra={
        "dynamic_check": DynamicCheck(
            expr=if_(ref("training.finetuning.pretraining_enabled"), exists(this()), True),
            message="pretraining_from must be specified if pretraining_enabled is True."
        )
    })
    pretraining_include_params: list[str] = Field(["model.*"])
    pretraining_exclude_params: list[str] = Field([])
    freezing_enabled: bool = Field(False)
    freezing_include_params: list[str] = Field([])
    freezing_exclude_params: list[str] = Field([])


class WeightAveragingConfig(ConfigBaseModel):
    ema_enabled: bool = Field(False)
    ema_decay: float = Field(0.999, gt=0, le=1)
    ema_include_params: list[str] = Field(["model.*"])
    ema_exclude_params: list[str] = Field([])


class SemiSupervisedConfig(ConfigBaseModel):
    enabled: bool = Field(False)
    aux_loss_weight: float = Field(0.25, ge=0.0)
    pseudo_frame_identity_weight: float = Field(0.25, ge=0.0)
    skip_penalty: float = Field(0.5, ge=0.0)
    min_agreement: float = Field(0.5, ge=0.0, le=1.0)
    min_confidence: float = Field(0.10)
    min_determinacy: float = Field(0.55, ge=0.0, le=1.0)
    min_monotonicity: float = Field(0.55, ge=0.0, le=1.0)


class TrainingConfig(ConfigBaseModel):
    augmentation: AugmentationConfig = Field(...)
    semisupervised: SemiSupervisedConfig = Field(default_factory=SemiSupervisedConfig)
    iterative_ranking: IterativeRankingConfig = Field(default_factory=IterativeRankingConfig)
    loss: LossConfig = Field(...)
    dataloader: DataLoaderConfig = Field(...)
    optimizer: OptimizerConfig = Field(...)
    lr_scheduler: LRSchedulerConfig = Field(...)
    trainer: TrainerConfig = Field(...)
    validation: ValidationConfig = Field(...)
    finetuning: FinetuningConfig = Field(...)
    weight_averaging: WeightAveragingConfig = Field(...)


class InferenceConfig(ConfigBaseModel):
    g2p: G2PPipelineConfig | None = Field(None, json_schema_extra={
        "dynamic_expr": ref("binarizer.g2p")
    })
    features: BinarizerFeaturesConfig = Field(None, json_schema_extra={
        "dynamic_expr": ref("binarizer.features")
    })


class RootConfig(ConfigBaseModel):
    binarizer: BinarizerConfig = Field(...)
    model: ModelConfig = Field(...)
    training: TrainingConfig = Field(...)
    inference: InferenceConfig = Field(...)
