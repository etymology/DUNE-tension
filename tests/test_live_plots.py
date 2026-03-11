from __future__ import annotations

from types import SimpleNamespace
from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dune_tension.gui.live_plots import LivePlotManager


def test_build_audio_diagnostics_figure_includes_fft_and_pesto_axes() -> None:
    waveform = np.sin(np.linspace(0.0, 8.0 * np.pi, 2048, dtype=np.float32))
    analysis = SimpleNamespace(
        activation_map=np.ones((32, 8), dtype=np.float32),
        activation_freq_axis=np.geomspace(40.0, 400.0, 32).astype(np.float32),
        frame_times=np.linspace(0.0, 0.04, 8, dtype=np.float32),
        predicted_frequencies=np.linspace(80.0, 120.0, 8, dtype=np.float32),
    )

    figure = LivePlotManager._build_audio_diagnostics_figure(
        waveform,
        8000,
        analysis,
    )

    assert len(figure.axes) == 3
    assert [axis.get_title() for axis in figure.axes] == [
        "Latest Captured Waveform",
        "FFT",
        "PESTO Activations",
    ]
    assert figure.axes[1].get_xlim()[1] <= 2000.0
    assert figure.axes[2].get_ylim() == (30.0, 400.0)


def test_pesto_axis_uses_99_percent_activation_cutoff() -> None:
    frequencies = np.array([40.0, 60.0, 80.0, 100.0], dtype=np.float32)
    activation = np.array(
        [
            [90.0, 90.0],
            [5.0, 5.0],
            [3.0, 3.0],
            [2.0, 2.0],
        ],
        dtype=np.float32,
    )

    cutoff = LivePlotManager._activation_coverage_max_frequency(
        frequencies,
        activation,
        coverage=0.99,
        fallback_max=2000.0,
    )

    assert cutoff == 100.0
