"""Configuration helpers for pitch comparison workflows."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np


@dataclasses.dataclass
class PitchCompareConfig:
    """Configuration loaded from JSON for pitch comparison runs."""

    sample_rate: int = 44100
    noise_duration: float = 2.0
    snr_threshold_db: float = 2.0
    min_frequency: float = 30.0
    max_frequency: float = 2000.0
    min_oscillations_per_window: float = 10.0
    min_window_overlap: float = 0.5
    idle_timeout: float = 0.2
    max_record_seconds: float = 10.0
    input_mode: str = "mic"
    input_audio_path: Optional[str] = None
    noise_audio_path: Optional[str] = None
    output_directory: str = "data"
    show_plots: bool = True
    show_pitch_overlay: bool = False
    crepe_model_capacity: str = "tiny"
    crepe_step_size_ms: Optional[float] = None
    over_subtraction: float = 1.0  # Noise reduction factor
    expected_f0: Optional[float] = None
    crepe_activation_coverage: float = 0.9
    pesto_model_name: str = "mir-1k_g7"
    pesto_step_size_ms: Optional[float] = None
    comb_trigger_on_rmax: float = 0.25
    comb_trigger_off_rmax: float = 0.18
    comb_trigger_sfm_max: float = 0.6
    comb_trigger_on_frames: int = 3
    comb_trigger_off_frames: int = 2
    comb_trigger_min_harmonics: int = 4

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "PitchCompareConfig":
        normalized = dict(raw)

        if "expected_f0" not in normalized and "sr_augment_factor" in normalized:
            try:
                sr_factor = float(normalized["sr_augment_factor"])
            except (TypeError, ValueError):
                sr_factor = float("nan")
            if np.isfinite(sr_factor) and sr_factor > 0:
                normalized["expected_f0"] = 1000.0 / sr_factor

        known = {f.name for f in dataclasses.fields(PitchCompareConfig)}
        filtered = {k: v for k, v in normalized.items() if k in known}
        return PitchCompareConfig(**filtered)


def load_config(path: Path) -> PitchCompareConfig:
    data = json.loads(path.read_text())
    return PitchCompareConfig.from_dict(data)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
