from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spectrum_analysis import crepe_analysis


def test_activation_to_frequency_confidence_time_weighting(monkeypatch):
    monkeypatch.setattr(crepe_analysis, "crepe", None)

    activation = np.array(
        [
            [0.2, 0.8],
            [0.4, 0.6],
        ],
        dtype=np.float32,
    )
    times = np.array([0.0, 0.1], dtype=np.float32)
    freq_axis = np.array([100.0, 200.0], dtype=np.float32)

    freq, confidence = crepe_analysis.activation_to_frequency_confidence(
        activation, times, freq_axis
    )

    expected_confidence = (0.8 * 0.1) + (0.6 * 0.1)
    assert np.isfinite(freq)
    assert np.isclose(confidence, expected_confidence)

    weighted_bins = np.array([0.02 + 0.04, 0.08 + 0.06])
    expected_freq = np.dot(freq_axis, weighted_bins) / weighted_bins.sum()
    assert np.isclose(freq, expected_freq)


def test_activation_to_frequency_confidence_last_duration(monkeypatch):
    monkeypatch.setattr(crepe_analysis, "crepe", None)

    activation = np.array(
        [
            [0.0, 0.0],
            [0.1, 0.9],
            [0.3, 0.7],
        ],
        dtype=np.float32,
    )
    times = np.array([0.0, 0.05, 0.15], dtype=np.float32)
    freq_axis = np.array([100.0, 200.0], dtype=np.float32)

    _, confidence = crepe_analysis.activation_to_frequency_confidence(
        activation, times, freq_axis
    )

    expected_confidence = (0.9 * 0.05) + (0.7 * 0.1)
    assert np.isclose(confidence, expected_confidence)
