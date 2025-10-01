"""Spectrum analysis utilities and interactive visualizer."""

from __future__ import annotations

from importlib import import_module
from typing import TYPE_CHECKING, Any

__all__ = [
    "AudioSource",
    "DemoSource",
    "MicSource",
    "ScrollingSpectrogram",
    "SpectrogramConfig",
    "EPS",
    "dbfs",
    "hann_window",
    "parse_args",
    "build_config",
    "create_source",
    "main",
    "compare_pitch_main",
    "estimate_pitch_from_audio",
    "listen_for_trigger_and_classify",
]

_EXPORT_MAP = {
    "AudioSource": ("spectrum_analysis.audio_sources", "AudioSource"),
    "DemoSource": ("spectrum_analysis.audio_sources", "DemoSource"),
    "MicSource": ("spectrum_analysis.audio_sources", "MicSource"),
    "ScrollingSpectrogram": (
        "spectrum_analysis.visualizer",
        "ScrollingSpectrogram",
    ),
    "SpectrogramConfig": (
        "spectrum_analysis.visualizer",
        "SpectrogramConfig",
    ),
    "EPS": ("spectrum_analysis.utils", "EPS"),
    "dbfs": ("spectrum_analysis.utils", "dbfs"),
    "hann_window": ("spectrum_analysis.utils", "hann_window"),
    "parse_args": ("spectrum_analysis.cli", "parse_args"),
    "build_config": ("spectrum_analysis.cli", "build_config"),
    "create_source": ("spectrum_analysis.cli", "create_source"),
    "main": ("spectrum_analysis.cli", "main"),
    "compare_pitch_main": (
        "spectrum_analysis.compare_pitch_cli",
        "main",
    ),
    "estimate_pitch_from_audio": (
        "spectrum_analysis.crepe_analysis",
        "estimate_pitch_from_audio",
    ),
    "listen_for_trigger_and_classify": (
        "spectrum_analysis.workflow",
        "listen_for_trigger_and_classify",
    ),
}


if TYPE_CHECKING:  # pragma: no cover - only for type checkers
    from spectrum_analysis.audio_sources import AudioSource, DemoSource, MicSource
    from spectrum_analysis.cli import build_config, create_source, main, parse_args
    from spectrum_analysis.compare_pitch_cli import main as compare_pitch_main
    from spectrum_analysis.crepe_analysis import estimate_pitch_from_audio
    from spectrum_analysis.utils import EPS, dbfs, hann_window
    from spectrum_analysis.visualizer import ScrollingSpectrogram, SpectrogramConfig
    from spectrum_analysis.workflow import listen_for_trigger_and_classify


def __getattr__(name: str) -> Any:
    """Lazily import heavy submodules on demand."""

    if name in _EXPORT_MAP:
        module_name, attribute = _EXPORT_MAP[name]
        module = import_module(module_name)
        value = getattr(module, attribute)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(set(__all__))
