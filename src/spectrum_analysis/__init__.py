"""Spectrum analysis utilities and interactive visualizer."""

from spectrum_analysis.audio_sources import AudioSource, DemoSource, MicSource
from spectrum_analysis.cli import build_config, create_source, main, parse_args
from spectrum_analysis.compare_pitch_cli import main as compare_pitch_main
from spectrum_analysis.crepe_analysis import estimate_pitch_from_audio
from spectrum_analysis.utils import EPS, dbfs, hann_window
from spectrum_analysis.visualizer import ScrollingSpectrogram, SpectrogramConfig
from spectrum_analysis.workflow import listen_for_trigger_and_classify

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
