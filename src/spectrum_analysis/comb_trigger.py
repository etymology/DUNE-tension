"""Harmonic comb trigger bridge backed by the Rust implementation."""

from __future__ import annotations

import asyncio
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

try:
    from harmonic_comb import (
        record_with_harmonic_comb as _rust_record_with_harmonic_comb,
        record_with_snr_trigger as _rust_record_with_snr_trigger,
    )
except ModuleNotFoundError as exc:  # pragma: no cover - extension must be built
    raise ImportError(
        "The harmonic_comb extension module is not available. "
        "Build the Rust extension (e.g., via `maturin develop`) before using the comb trigger."
    ) from exc


@dataclass(slots=True)
class HarmonicCombConfig:
    """Runtime configuration for the harmonic comb trigger."""

    frame_size: int = 2048
    hop_size: int = 1024
    candidate_count: int = 36
    harmonic_weight_count: int = 10
    min_harmonics: int = 4
    on_rmax: float = 0.001
    off_rmax: float = 0.0005
    sfm_max: float = 0.6
    on_frames: int = 3
    off_frames: int = 3

    def harmonic_weights(self) -> NDArray[np.float64]:
        """Return per-harmonic weights used when scoring candidates."""

        count = max(1, int(self.harmonic_weight_count))
        return 1.0 / np.arange(1, count + 1, dtype=np.float64)

    def to_mapping(self) -> dict[str, Any]:
        """Return a plain dictionary representing the configuration."""

        return asdict(self)


@dataclass(slots=True)
class SnrTriggerConfig:
    """Runtime configuration for the RMS-based SNR trigger."""

    hop_size: int = 1024
    snr_threshold_db: float = 2.0
    idle_timeout: float = 0.2

    def to_mapping(self) -> dict[str, Any]:
        """Return a plain dictionary representing the configuration."""

        return asdict(self)


async def record_with_harmonic_comb_async(
    *,
    expected_f0: float,
    sample_rate: int,
    max_record_seconds: float,
    comb_cfg: HarmonicCombConfig,
) -> NDArray[np.float32]:
    """Record audio using the harmonic comb trigger asynchronously."""

    cfg_mapping = comb_cfg.to_mapping()
    result = await _rust_record_with_harmonic_comb(
        expected_f0,
        int(sample_rate),
        float(max_record_seconds),
        cfg_mapping,
    )
    return np.asarray(result, dtype=np.float32)


async def record_with_snr_trigger_async(
    *,
    sample_rate: int,
    max_record_seconds: float,
    noise_rms: float,
    snr_cfg: SnrTriggerConfig,
) -> NDArray[np.float32]:
    """Record audio using an RMS-based SNR trigger asynchronously."""

    cfg_mapping = snr_cfg.to_mapping()
    result = await _rust_record_with_snr_trigger(
        int(sample_rate),
        float(max_record_seconds),
        float(noise_rms),
        cfg_mapping,
    )
    return np.asarray(result, dtype=np.float32)


def record_with_harmonic_comb(
    *,
    expected_f0: float,
    sample_rate: int,
    max_record_seconds: float,
    comb_cfg: HarmonicCombConfig,
) -> NDArray[np.float32]:
    """Record audio using the harmonic comb trigger."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            record_with_harmonic_comb_async(
                expected_f0=expected_f0,
                sample_rate=sample_rate,
                max_record_seconds=max_record_seconds,
                comb_cfg=comb_cfg,
            )
        )

    raise RuntimeError(
        "record_with_harmonic_comb cannot be used while an event loop is running. "
        "Use record_with_harmonic_comb_async instead."
    )


def record_with_snr_trigger(
    *,
    sample_rate: int,
    max_record_seconds: float,
    noise_rms: float,
    snr_cfg: SnrTriggerConfig,
) -> NDArray[np.float32]:
    """Record audio using an RMS-based SNR trigger."""

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            record_with_snr_trigger_async(
                sample_rate=sample_rate,
                max_record_seconds=max_record_seconds,
                noise_rms=noise_rms,
                snr_cfg=snr_cfg,
            )
        )

    raise RuntimeError(
        "record_with_snr_trigger cannot be used while an event loop is running. "
        "Use record_with_snr_trigger_async instead."
    )


__all__ = [
    "HarmonicCombConfig",
    "record_with_harmonic_comb",
    "record_with_harmonic_comb_async",
    "SnrTriggerConfig",
    "record_with_snr_trigger",
    "record_with_snr_trigger_async",
]
