"""Spectrum analysis utilities and interactive visualizer."""

from .audio_sources import AudioSource, DemoSource, MicSource
from .cli import build_config, create_source, main, parse_args
from .compare_pitch_cli import main as compare_pitch_main
from .crepe_analysis import estimate_pitch_from_audio
from .utils import EPS, dbfs, hann_window
from .visualizer import ScrollingSpectrogram, SpectrogramConfig
from .workflow import listen_for_trigger_and_classify

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
