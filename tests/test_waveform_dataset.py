from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dune_tension.waveform_dataset import WaveformParameters, generate_waveform


def _params(**overrides: float | int | str) -> WaveformParameters:
    base = dict(
        sample_rate=8000,
        duration=1.0,
        base_frequency=220.0,
        waveform="square",
        num_partials=4,
        gain=0.8,
        noise_level=0.0,
        spectral_tilt_db_per_octave=0.0,
        partial_decay_bias=0.0,
        seed=1234,
        normalize=False,
    )
    base.update(overrides)
    return WaveformParameters(**base)


def test_waveform_reaches_noise_floor_at_end():
    params = _params(noise_level=0.05)
    result = generate_waveform(params, include_noise=False, return_envelopes=True)
    # The tail of the deterministic signal should be nearly silent.
    tail = result.deterministic[int(0.99 * len(result.deterministic)) :]
    assert np.max(np.abs(tail)) < 1e-3


def test_positive_bias_speeds_high_partials():
    params = _params(partial_decay_bias=2.5)
    result = generate_waveform(params, include_noise=False, return_envelopes=True)
    envelopes = [
        env for env in result.envelopes if env.size and not np.allclose(env, 0.0)
    ]
    mid_index = len(result.deterministic) // 2
    low_envelope = envelopes[0]
    high_envelope = envelopes[-1]
    low_ratio = np.abs(low_envelope[mid_index]) / np.abs(low_envelope[0])
    high_ratio = np.abs(high_envelope[mid_index]) / np.abs(high_envelope[0])
    assert high_ratio < low_ratio


def test_negative_bias_slows_high_partials():
    params = _params(partial_decay_bias=-2.5)
    result = generate_waveform(params, include_noise=False, return_envelopes=True)
    envelopes = [
        env for env in result.envelopes if env.size and not np.allclose(env, 0.0)
    ]
    mid_index = len(result.deterministic) // 2
    low_envelope = envelopes[0]
    high_envelope = envelopes[-1]
    low_ratio = np.abs(low_envelope[mid_index]) / np.abs(low_envelope[0])
    high_ratio = np.abs(high_envelope[mid_index]) / np.abs(high_envelope[0])
    assert high_ratio > low_ratio
