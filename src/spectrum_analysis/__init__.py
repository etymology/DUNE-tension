"""Spectrum analysis utilities and interactive visualizer."""
from .audio import AudioSource, DemoSource, MicSource
from .cli import build_config, create_source, main, parse_args
from .utils import EPS, dbfs, hann_window
from .visualizer import ScrollingSpectrogram, SpectrogramConfig

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
]
