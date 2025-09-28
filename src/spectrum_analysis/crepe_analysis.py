"""Helpers for working with CREPE activations."""

from __future__ import annotations

from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
from numpy.lib.stride_tricks import as_strided

from audio_processing import determine_window_and_hop

CREPE_FRAME_TARGET_RMS = 0.5

try:  # Optional dependency - heavy ML models
    import crepe  # type: ignore
except Exception:  # pragma: no cover - dependency may be absent
    crepe = None  # type: ignore

if TYPE_CHECKING:  # pragma: no cover - only for type checking
    from compare_pitch_cli import PitchCompareConfig


def compute_crepe_activation(
    audio: np.ndarray,
    cfg: "PitchCompareConfig",
    sr_augment_factor: Optional[float] = None,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Compute CREPE activations for a signal, with optional preprocessing augment of sample rate."""

    if crepe is None:
        print("[WARN] crepe is not installed; skipping CREPE activation plot.")
        return None

    crepe_sr = (
        cfg.sample_rate
        if sr_augment_factor is None
        else cfg.sample_rate * sr_augment_factor
    )
    if not np.isfinite(crepe_sr) or crepe_sr <= 0:
        raise ValueError("CREPE sample rate must be positive and finite.")

    window_samples, hop_samples = determine_window_and_hop(cfg, len(audio))
    if cfg.crepe_step_size_ms is not None:
        step_ms = float(cfg.crepe_step_size_ms)
    else:
        step_sec = hop_samples / crepe_sr
        step_ms = max(step_sec * 1000.0, 1.0)

    min_overlap = float(np.clip(cfg.min_window_overlap, 0.0, 0.999))
    max_step_fraction = max(1.0 - min_overlap, 0.0)
    max_step_ms = max((window_samples * max_step_fraction) / crepe_sr * 1000.0, 1.0)
    step_ms = float(np.clip(step_ms, 1.0, max_step_ms))

    activation = _get_activation_with_frame_gain(
        audio,
        int(round(crepe_sr)),
        model_capacity=cfg.crepe_model_capacity,
        center=True,
        step_size=int(round(step_ms)),
        verbose=True,
    )

    if sr_augment_factor is not None and sr_augment_factor != 1.0:
        activation = _reverse_sr_augment(activation, sr_augment_factor)

    frame_count = activation.shape[0]
    crepe_times = np.arange(frame_count) * step_ms / 1000.0
    if sr_augment_factor is not None:
        crepe_times = crepe_times * sr_augment_factor
    return (
        np.asarray(crepe_times, dtype=np.float32),
        np.asarray(activation, dtype=np.float32),
    )


def _reverse_sr_augment(activation: np.ndarray, sr_augment_factor: float) -> np.ndarray:
    """Shift CREPE activation bins to account for scaled sample rate."""

    num_bins = activation.shape[1]
    bin_shift = int(round(-np.log2(sr_augment_factor) * 60.0))
    if abs(bin_shift) < num_bins:
        if bin_shift > 0:
            activation = np.pad(activation, ((0, 0), (bin_shift, 0)), mode="constant")[
                :, :num_bins
            ]
        elif bin_shift < 0:
            activation = np.pad(activation, ((0, 0), (0, -bin_shift)), mode="constant")[
                :, -bin_shift:
            ]
    else:
        activation = np.zeros_like(activation)
    return activation


def summarize_activation(activation: np.ndarray) -> Tuple[float, float]:
    """Return fundamental frequency and confidence summary for an activation map."""

    if crepe is None:
        return float("nan"), float("nan")

    if activation.size == 0:
        return float("nan"), float("nan")

    voiced_mask = activation.max(axis=1) > 0
    if not np.any(voiced_mask):
        return float("nan"), float("nan")

    average_activations = activation[voiced_mask].mean(axis=0, keepdims=True)
    cents = crepe.core.to_local_average_cents(average_activations)
    frequency = 10 * 2 ** (cents / 1200.0)
    confidence = average_activations.max(axis=1)

    freq_value = float(np.squeeze(frequency))
    conf_value = float(np.squeeze(confidence))
    return freq_value, conf_value


def crepe_frequency_axis(num_bins: int) -> np.ndarray:
    """Return the frequency axis (Hz) for CREPE activations."""

    base_freq = 32.703195662574764  # C1 in Hz; matches CREPE/PESTO documentation
    bins_per_octave = 60.0  # 20 cents per bin
    return base_freq * (2.0 ** (np.arange(num_bins) / bins_per_octave))


def _get_activation_with_frame_gain(
    audio: np.ndarray,
    sr: int,
    *,
    model_capacity: str = "full",
    center: bool = True,
    step_size: int = 10,
    verbose: int = 1,
    target_rms: float = CREPE_FRAME_TARGET_RMS,
) -> np.ndarray:
    """Copy of :func:`crepe.core.get_activation` with per-frame RMS gain."""

    if crepe is None:  # pragma: no cover - handled by caller
        raise RuntimeError("CREPE is not available")

    model = crepe.core.build_and_load_model(model_capacity)
    model_srate = crepe.core.model_srate

    if audio.ndim == 2:
        audio = audio.mean(axis=1)
    audio = audio.astype(np.float32)

    if sr != model_srate:
        from resampy import resample  # type: ignore

        audio = resample(audio, sr, model_srate)
        sr = model_srate

    if center:
        audio = np.pad(audio, 512, mode="constant", constant_values=0)

    hop_length = int(model_srate * step_size / 1000)
    n_frames = 1 + int((len(audio) - 1024) / hop_length)
    frames = as_strided(
        audio,
        shape=(1024, n_frames),
        strides=(audio.itemsize, hop_length * audio.itemsize),
    )
    frames = frames.transpose().copy()

    frame_means = np.mean(frames, axis=1, keepdims=True)
    frames -= frame_means
    frame_stds = np.std(frames, axis=1, keepdims=True)
    frame_stds = np.clip(frame_stds, 1e-8, None)
    frames /= frame_stds

    if target_rms > 0.0:
        frames *= np.float32(target_rms)

    frames = np.asarray(frames, dtype=np.float32)

    return model.predict(frames, verbose=verbose)
