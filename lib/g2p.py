"""G2P construction with TIFA's local plugins registered."""

from pathlib import Path

from g2pflow import build_pipeline_from_config, register_plugin_paths

__all__ = ["build_pipeline_from_config"]

# Register on import so spawned workers can unpickle plugin instances.
register_plugin_paths(Path(__file__).resolve().parents[1] / "plugins" / "g2p")
