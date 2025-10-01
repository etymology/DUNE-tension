"""Utilities for locating the y-position that maximizes microphone RMS."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
import math
import time
from typing import Sequence

import numpy as np

from . import plc_io
from .audioProcessing import get_samplerate, record_audio_filtered

MicrophoneSampler = Callable[
    [float, int], tuple[Sequence[float] | np.ndarray | None, float]
]


@dataclass(slots=True)
class ScanConfig:
    """Configuration controlling how the calibration scan is performed."""

    step: float = 0.1
    """Step size, in millimetres, used while traversing the search range."""

    settle_time: float = 0.05
    """Delay, in seconds, to wait after each move before sampling audio."""

    sample_duration: float = 0.1
    """Duration, in seconds, of each microphone capture."""

    speed: float | None = 150.0
    """Optional override for the Y-axis scan speed."""


def _measure_microphone_rms(
    sampler: MicrophoneSampler,
    duration: float,
    sample_rate: int,
) -> float | None:
    """Record audio using *sampler* and return the RMS amplitude."""

    audio, _ = sampler(duration, sample_rate)
    if audio is None:
        return None

    buffer = np.asarray(audio, dtype=np.float64)
    if buffer.size == 0:
        return None
    return float(np.sqrt(np.mean(np.square(buffer))))


def _fit_gaussian_peak(
    positions: Sequence[float], rms_values: Sequence[float]
) -> float | None:
    """Estimate the peak position assuming a Gaussian RMS profile."""

    if len(positions) < 3:
        return None

    positions_arr = np.asarray(positions, dtype=np.float64)
    rms_arr = np.asarray(rms_values, dtype=np.float64)

    if np.any(rms_arr <= 0):
        return None

    log_rms = np.log(rms_arr)
    coeffs = np.polyfit(positions_arr, log_rms, deg=2)
    a, b, _ = coeffs

    if not math.isfinite(a) or not math.isfinite(b) or a >= 0:
        return None

    return float(-b / (2 * a))


def calibrate(
    dy: float,
    *,
    config: ScanConfig | None = None,
    sampler: MicrophoneSampler | None = None,
    sample_rate: int | None = None,
) -> float:
    """Return the y-coordinate that maximizes the microphone RMS response."""

    if dy <= 0:
        msg = "dy must be positive to perform a calibration scan."
        raise ValueError(msg)

    cfg = config or ScanConfig()
    if cfg.step <= 0:
        msg = "Scan step must be positive."
        raise ValueError(msg)

    x_start, y_start = plc_io.get_xy()
    y_lower = y_start - dy / 2

    goto_kwargs = {"speed": cfg.speed} if cfg.speed is not None else {}
    plc_io.goto_xy(x_start, y_lower, **goto_kwargs)

    if sampler is None:
        sampler = record_audio_filtered
        sample_rate_val = sample_rate or get_samplerate()
        if sample_rate_val is None:
            msg = "Unable to determine the microphone sample rate."
            raise RuntimeError(msg)
        sample_rate = int(sample_rate_val)
    elif sample_rate is None:
        msg = "sample_rate must be provided when using a custom sampler."
        raise ValueError(msg)
    else:
        sample_rate = int(sample_rate)

    num_steps = max(1, round(dy / cfg.step))
    actual_step = dy / num_steps

    positions: list[float] = []
    rms_values: list[float] = []

    for i in range(num_steps + 1):
        y_target = y_lower + actual_step * i
        plc_io.goto_xy(x_start, y_target, **goto_kwargs)

        if cfg.settle_time > 0:
            time.sleep(cfg.settle_time)

        y_actual = plc_io.get_cached_xy()[1]
        rms = _measure_microphone_rms(sampler, cfg.sample_duration, sample_rate)
        if rms is None:
            continue

        positions.append(y_actual)
        rms_values.append(rms)

    if not positions:
        msg = "No microphone samples were acquired during calibration."
        raise RuntimeError(msg)

    best_index = int(np.argmax(rms_values))
    peak = _fit_gaussian_peak(positions, rms_values)
    if peak is None or peak < min(positions) or peak > max(positions):
        peak = positions[best_index]

    plc_io.goto_xy(x_start, peak, **goto_kwargs)
    return peak
