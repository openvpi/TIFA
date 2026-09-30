import pathlib

import lightning.pytorch as pl
import lightning.pytorch.callbacks
import torch
from g2pflow import G2PPipelineConfig
from lightning_utilities.core.rank_zero import rank_zero_only, rank_zero_info
from torch import Tensor

from lib import logging
from lib.config.core import ConfigBaseModel
from lib.config.formatter import format_model
from lib.config.io import load_raw_config
from lib.config.schema import (
    InferenceConfig,
    ModelConfig,
)
from lib.vocabulary import Vocabulary
from .backend import (
    ForcedAlignmentInferenceModel,
    ForcedAlignmentSSLInferenceModel,
    InferenceBackend,
)
from .data import PairedDataset
from .module import ForcedAlignmentInferenceModule, OfflineEvaluationModule

__all__ = [
    "load_config_for_inference",
    "load_g2p_config",
    "load_state_dict_for_inference",
    "load_inference_model",
    "run_inference",
    "evaluate_offline",
]


@rank_zero_only
def _log_config(cfg: ConfigBaseModel):
    print(format_model(cfg))


def load_g2p_config(path: pathlib.Path) -> G2PPipelineConfig:
    raw = load_raw_config(path, inherit=False)
    return G2PPipelineConfig.model_validate(raw)


def load_config_for_inference(
        path: pathlib.Path,
        scope: int = 0
) -> tuple[ModelConfig, InferenceConfig]:
    if not path.is_file():
        raise FileNotFoundError(f"Config file not found: {path}")
    config = load_raw_config(path, inherit=True, overrides=None)
    model_config = ModelConfig.model_validate(config["model"], scope=scope)
    inference_config = InferenceConfig.model_validate(config["inference"], scope=scope)
    model_config.check(scope_mask=scope)
    inference_config.check(scope_mask=scope)

    _log_config(model_config)
    _log_config(inference_config)

    return model_config, inference_config


def load_state_dict_for_inference(path: pathlib.Path) -> dict[str, Tensor]:
    checkpoint = torch.load(path, map_location="cpu")
    state_dict: dict = checkpoint.get("state_dict", {})
    if not state_dict:
        raise KeyError(f"No valid state dict found in checkpoint: {path}.")
    return state_dict


_ARCH_BACKEND_MAP: dict[str, type[InferenceBackend]] = {
    "ForcedAlignmentModel": ForcedAlignmentInferenceModel,
    "ForcedAlignmentSSLModel": ForcedAlignmentSSLInferenceModel,
}


def load_inference_model(
    checkpoint_path: str | pathlib.Path,
    scope: int = 0,
) -> tuple[InferenceBackend, Vocabulary, InferenceConfig]:
    """Load an InferenceBackend, vocabulary, and inference configuration.

    Args:
        checkpoint_path: Path to the .ckpt file.
        scope: ConfigurationScope value.

    Returns:
        (backend, vocabulary, inference_config)
    """
    checkpoint_path = pathlib.Path(checkpoint_path)
    config_path = checkpoint_path.parent / "config.yaml"

    model_config, inference_config = load_config_for_inference(
        config_path, scope=scope,
    )

    # Build backend
    arch = model_config.arch
    backend_cls: type[ForcedAlignmentInferenceModel | ForcedAlignmentSSLInferenceModel] = _ARCH_BACKEND_MAP.get(arch)
    if backend_cls is None:
        raise ValueError(
            f"Unknown model architecture: '{arch}'. "
            f"Expected one of: {list(_ARCH_BACKEND_MAP)}."
        )

    # Load vocabulary
    vocab_path = checkpoint_path.parent / "vocabulary.json"
    if not vocab_path.is_file():
        raise FileNotFoundError(
            f"Vocabulary not found at {vocab_path}"
        )
    vocabulary = Vocabulary.from_file(vocab_path)

    backend = backend_cls(model_config, inference_config)

    # Load state dict
    state_dict = load_state_dict_for_inference(checkpoint_path)
    backend.load_state_dict(state_dict, strict=True)
    backend.eval()

    logging.info(
        f"Loaded model from '{checkpoint_path}'.", callback=rank_zero_info,
    )

    return backend, vocabulary, inference_config


def run_inference(
    backend: InferenceBackend,
    dataset: torch.utils.data.Dataset,
    callbacks: list[lightning.pytorch.callbacks.Callback],
    batch_size: int = 1,
    num_workers: int = 0,
    precision: str = "32-true",
    mode: str = "predict",
    score_unit: str = "levenshtein",
    skip_penalty: float = 0.5,
) -> None:
    """Run inference (predict) or online evaluation (evaluate)."""
    module = ForcedAlignmentInferenceModule(backend, score_unit=score_unit, skip_penalty=skip_penalty)
    trainer = pl.Trainer(
        precision=precision,
        logger=False,
        enable_checkpointing=False,
        callbacks=callbacks,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        prefetch_factor=2 if num_workers > 0 else None,
        shuffle=False,
        persistent_workers=num_workers > 0,
        collate_fn=dataset.collate if hasattr(dataset, "collate") else None,
    )
    if mode == "predict":
        trainer.predict(module, dataloader)
    elif mode == "evaluate":
        trainer.test(module, dataloader)
    else:
        raise ValueError(f"Unknown mode: {mode}")


def evaluate_offline(
    dataset: PairedDataset,
    callbacks: list[lightning.pytorch.callbacks.Callback],
) -> None:
    """Offline evaluation via trainer.test() with OfflineEvaluationModule.

    Batch size and num_workers are hard-coded to 1/0 -- no benefit
    from batching pre-computed text data.
    """
    module = OfflineEvaluationModule()
    trainer = pl.Trainer(
        precision="32-true",
        logger=False,
        enable_checkpointing=False,
        callbacks=callbacks,
    )
    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=1,
        num_workers=0,
        shuffle=False,
        collate_fn=dataset.collate if hasattr(dataset, "collate") else None,
    )
    trainer.test(module, dataloader)
