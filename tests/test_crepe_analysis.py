from pathlib import Path
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from spectrum_analysis import crepe_analysis


def test_activations_to_pitch_time_weighting(monkeypatch):
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

    freq, confidence = crepe_analysis.activations_to_pitch(activation, times, freq_axis)

    expected_confidence = (0.8 * 0.1) + (0.6 * 0.1)
    assert np.isfinite(freq)
    assert np.isclose(confidence, expected_confidence)

    weighted_bins = np.array([0.02 + 0.04, 0.08 + 0.06])
    expected_freq = np.dot(freq_axis, weighted_bins) / weighted_bins.sum()
    assert np.isclose(freq, expected_freq)


def test_activations_to_pitch_last_duration(monkeypatch):
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

    _, confidence = crepe_analysis.activations_to_pitch(activation, times, freq_axis)

    expected_confidence = (0.9 * 0.05) + (0.7 * 0.1)
    assert np.isclose(confidence, expected_confidence)


def test_activations_to_pitch_expected_frequency_mask(monkeypatch):
    monkeypatch.setattr(crepe_analysis, "crepe", None)

    activation = np.array(
        [
            [0.1, 0.2, 5.0],
            [0.2, 0.3, 4.0],
        ],
        dtype=np.float32,
    )
    times = np.array([0.0, 0.1], dtype=np.float32)
    freq_axis = np.array([100.0, 200.0, 400.0], dtype=np.float32)

    freq, confidence = crepe_analysis.activations_to_pitch(
        activation,
        times,
        freq_axis,
        expected_frequency=120.0,
    )

    assert np.isclose(freq, 162.5)
    expected_confidence = (0.2 * 0.1) + (0.3 * 0.1)
    assert np.isclose(confidence, expected_confidence)


def test_estimate_pitch_from_audio(monkeypatch):
    monkeypatch.setattr(crepe_analysis, "crepe", None)

    captured = {}

    def fake_compute(audio, cfg, sr_augment_factor=None):
        captured["sample_rate"] = cfg.sample_rate
        captured["expected_f0"] = cfg.expected_f0
        return (
            np.array([0.0, 0.1], dtype=np.float32),
            np.array([[0.1, 0.9], [0.2, 0.8]], dtype=np.float32),
        )

    monkeypatch.setattr(crepe_analysis, "compute_crepe_activation", fake_compute)
    monkeypatch.setattr(
        crepe_analysis,
        "crepe_frequency_axis",
        lambda num_bins: np.array([100.0, 300.0], dtype=np.float32),
    )

    freq, confidence = crepe_analysis.estimate_pitch_from_audio(
        np.zeros(8, dtype=np.float32),
        sample_rate=16000,
        expected_frequency=120.0,
    )

    assert captured["sample_rate"] == 16000
    assert captured["expected_f0"] == 120.0
    assert np.isclose(freq, 100.0)
    expected_confidence = (0.1 * 0.1) + (0.2 * 0.1)
    assert np.isclose(confidence, expected_confidence)
