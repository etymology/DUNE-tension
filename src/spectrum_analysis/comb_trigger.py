"""Harmonic comb trigger utilities."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from audio_sources import MicSource, sd


@dataclass
class HarmonicCombConfig:
    """Runtime configuration for the harmonic comb trigger."""

    frame_size: int = 2048
    hop_size: int = 1024
    candidate_count: int = 36
    harmonic_weight_count: int = 10
    min_harmonics: int = 3
    on_rmax: float = 0.01
    off_rmax: float = 0.000001
    sfm_max: float = 0.6
    on_frames: int = 3
    off_frames: int = 2

    def harmonic_weights(self) -> np.ndarray:
        """Return per-harmonic weights used when scoring candidates."""

        count = max(1, int(self.harmonic_weight_count))
        return 1.0 / np.arange(1, count + 1, dtype=np.float64)


def _spectral_flatness(magnitude: np.ndarray) -> float:
    eps = 1e-12
    magnitude = np.maximum(magnitude, eps)
    geom_mean = np.exp(np.mean(np.log(magnitude)))
    arith_mean = np.mean(magnitude)
    return float(geom_mean / (arith_mean + eps))


def _harmonic_comb_response(
    frame: np.ndarray,
    sample_rate: int,
    window: np.ndarray,
    freq_bins: np.ndarray,
    candidates: np.ndarray,
    weights: np.ndarray,
    min_harmonics: int,
) -> tuple[float, float, bool]:
    windowed = frame * window
    spectrum = np.fft.rfft(windowed)
    magnitude = np.abs(spectrum)
    sfm = _spectral_flatness(magnitude)
    magnitude_db = 20.0 * np.log10(np.maximum(magnitude, 1e-12))

    max_mag = float(np.max(magnitude) + 1e-12)
    nyquist = sample_rate / 2.0
    bin_width = freq_bins[1] - freq_bins[0] if freq_bins.size > 1 else nyquist

    best_r = 0.0
    found = False

    for candidate in candidates:
        if not np.isfinite(candidate) or candidate <= 0.0:
            continue

        harmonics = candidate * np.arange(1, weights.size + 1, dtype=np.float64)
        valid_mask = harmonics <= nyquist
        if not np.any(valid_mask):
            continue

        harmonics = harmonics[valid_mask]
        local_weights = weights[: harmonics.size]

        sampled = np.interp(harmonics, freq_bins, magnitude, left=0.0, right=0.0)
        if sampled.size < min_harmonics:
            continue

        amps_db = 20.0 * np.log10(np.maximum(sampled, 1e-12))
        idx = np.clip(
            np.round(harmonics / max(bin_width, 1e-12)).astype(int),
            0,
            magnitude_db.size - 1,
        )
        prominences = []
        for bin_idx in idx:
            lo = max(bin_idx - 3, 0)
            hi = min(bin_idx + 3, magnitude_db.size - 1)
            prominences.append(magnitude_db[lo : hi + 1])
        if prominences:
            floor_db = np.array([float(np.median(p)) for p in prominences])
        else:
            floor_db = np.zeros(0, dtype=np.float64)

        if floor_db.size < sampled.size:
            floor_db = np.pad(floor_db, (0, sampled.size - floor_db.size), mode="edge")

        if np.count_nonzero(amps_db - floor_db >= 8.0) < min_harmonics:
            continue

        local_weight_sum = float(np.sum(local_weights))
        weighted_sum = float(np.sum(local_weights * sampled))
        if local_weight_sum > 0.0:
            r_value = weighted_sum / (local_weight_sum * max_mag)
        else:
            r_value = 0.0

        if r_value > best_r:
            best_r = r_value
            found = True

    return best_r, sfm, found


def record_with_harmonic_comb(
    *,
    expected_f0: float,
    sample_rate: int,
    max_record_seconds: float,
    comb_cfg: HarmonicCombConfig,
) -> np.ndarray:
    """Record audio using the harmonic comb trigger."""

    if sd is None:
        raise RuntimeError(
            "sounddevice is required for microphone recording but is not available."
        )

    frame_size = max(1, int(comb_cfg.frame_size))
    hop = max(1, int(comb_cfg.hop_size))
    window = np.hanning(frame_size).astype(np.float32)
    freq_bins = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    nyquist = sample_rate / 2.0

    f_min = max(
        expected_f0 / 2.0, freq_bins[1] if freq_bins.size > 1 else expected_f0 / 2.0
    )
    f_max = min(expected_f0 * 2.0, nyquist)
    if not np.isfinite(f_min) or not np.isfinite(f_max) or f_max <= f_min:
        raise ValueError("Invalid frequency band for harmonic comb trigger.")

    candidates = np.geomspace(f_min, f_max, num=max(1, int(comb_cfg.candidate_count)))
    weights = comb_cfg.harmonic_weights()
    min_harmonics = max(1, int(comb_cfg.min_harmonics))
    on_frames = max(1, int(comb_cfg.on_frames))
    off_frames = max(1, int(comb_cfg.off_frames))

    source = MicSource(sample_rate, hop)
    source.start()
    print("[INFO] Listening for audio events (harmonic comb trigger)...")

    collected: list[np.ndarray] = []
    max_samples = int(max_record_seconds * sample_rate)
    collected_samples = 0

    frame_buffer = np.zeros(0, dtype=np.float32)
    recent_chunks: deque[np.ndarray] = deque()
    recent_samples = 0

    on_counter = 0
    off_counter = 0
    triggered = False

    try:
        while collected_samples < max_samples:
            chunk = source.read()
            if chunk.size == 0:
                continue

            if chunk.dtype != np.float32:
                chunk = chunk.astype(np.float32, copy=False)

            frame_buffer = np.concatenate((frame_buffer, chunk))

            recent_chunks.append(chunk.copy())
            recent_samples += len(chunk)
            while recent_samples > frame_size:
                removed = recent_chunks.popleft()
                recent_samples -= len(removed)

            was_triggered = triggered
            chunk_included = False
            stop_recording = False

            while frame_buffer.size >= frame_size:
                frame = frame_buffer[:frame_size]
                r_value, sfm, valid = _harmonic_comb_response(
                    frame,
                    sample_rate,
                    window,
                    freq_bins,
                    candidates,
                    weights,
                    min_harmonics,
                )
                frame_buffer = frame_buffer[hop:]

                if not triggered:
                    if valid and r_value > comb_cfg.on_rmax and sfm < comb_cfg.sfm_max:
                        on_counter += 1
                    else:
                        on_counter = 0

                    if on_counter >= on_frames:
                        triggered = True
                        on_counter = 0
                        off_counter = 0
                        chunk_included = True
                        pre_audio = (
                            np.concatenate(list(recent_chunks))
                            if recent_chunks
                            else np.empty(0, dtype=np.float32)
                        )
                        if pre_audio.size:
                            collected.append(pre_audio.astype(np.float32, copy=False))
                            collected_samples += pre_audio.size
                        recent_chunks.clear()
                        recent_samples = 0
                        print("[INFO] Recording started (harmonic comb trigger).")
                        if collected_samples >= max_samples:
                            stop_recording = True
                            break
                else:
                    if r_value < comb_cfg.off_rmax:
                        off_counter += 1
                        if off_counter >= off_frames:
                            triggered = False
                            stop_recording = True
                            print("[INFO] Recording stopped (comb trigger released).")
                            break
                    else:
                        off_counter = 0

            if was_triggered and not chunk_included:
                collected.append(chunk.astype(np.float32, copy=False))
                collected_samples += len(chunk)

            if collected_samples >= max_samples:
                print("[WARN] Max recording length reached.")
                break

            if stop_recording:
                break
        else:
            print("[WARN] Max recording length reached.")
    finally:
        source.stop()

    if not collected:
        raise RuntimeError("No audio captured above the comb trigger thresholds.")

    return np.concatenate(collected).astype(np.float32)
