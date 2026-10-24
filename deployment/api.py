import json
import pathlib

from deployment.exporter import Exporter
from inference.api import load_config_for_inference
from inference.backend import InferenceBackend
from inference.module import VoiceEventInferenceModule
from lib import logging
from lib.config.schema import BinarizerFeaturesConfig, ConfigurationScope
from lib.vocabulary import Vocabulary
from modules.event_detection import VoiceEventDetectionModel


def _save_config(save_dir: pathlib.Path, features: BinarizerFeaturesConfig, **model_fields):
    config = {
        "samplerate": features.audio_sample_rate,
        "timestep": features.timestep,
        "hop_size": features.hop_size,
        "fft_size": features.fft_size,
        "win_size": features.win_size,
        "num_mels": features.spectrogram.num_bins,
        **model_fields,
    }
    with (save_dir / "config.json").open("w", encoding="utf8") as f:
        json.dump(config, f, ensure_ascii=False, indent=4)


def deploy_model(
    model: InferenceBackend,
    vocabulary: Vocabulary,
    save_dir: str | pathlib.Path,
    opset_version: int = 18,
    *,
    config_path: str | pathlib.Path,
):
    model_config, inference_config = load_config_for_inference(
        pathlib.Path(config_path), scope=ConfigurationScope.FA,
    )
    features = inference_config.features
    save_dir = pathlib.Path(save_dir)
    Exporter(model, save_dir, opset_version=opset_version).export()
    _save_config(save_dir, features, vocab_size=model_config.max_vocab_size)
    vocabulary.dump(save_dir / "vocabulary.json")
    logging.success(f"Deployment completed: '{save_dir.as_posix()}'.")


def deploy_voice_event_detector(
    model: VoiceEventDetectionModel,
    events: list[str],
    features: BinarizerFeaturesConfig,
    save_dir: str | pathlib.Path,
    opset_version: int = 18,
):
    save_dir = pathlib.Path(save_dir)
    module = VoiceEventInferenceModule(model, events, features)
    Exporter(module, save_dir, opset_version=opset_version).export()
    _save_config(save_dir, features, num_classes=model.num_classes)
    with (save_dir / "vocabulary.json").open("w", encoding="utf8") as f:
        json.dump({"events": events}, f, ensure_ascii=False, indent=4)
    logging.success(f"Deployment completed: '{save_dir.as_posix()}'.")
