"""Configuration helpers for pitch comparison workflows."""

from __future__ import annotations

import dataclasses
import json
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np

from spectrum_analysis.comb_trigger import HarmonicCombConfig


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
    max_record_seconds: float = 2.0
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
    comb_trigger: HarmonicCombConfig = dataclasses.field(
        default_factory=HarmonicCombConfig
    )

    @staticmethod
    def from_dict(raw: Dict[str, Any]) -> "PitchCompareConfig":
        normalized = dict(raw)

        comb_config_data: Dict[str, Any] = {}
        comb_raw = normalized.pop("comb_trigger", None)
        if isinstance(comb_raw, dict):
            comb_config_data.update(comb_raw)

        comb_aliases = {
            "comb_trigger_on_rmax": "on_rmax",
            "comb_trigger_off_rmax": "off_rmax",
            "comb_trigger_sfm_max": "sfm_max",
            "comb_trigger_on_frames": "on_frames",
            "comb_trigger_off_frames": "off_frames",
            "comb_trigger_min_harmonics": "min_harmonics",
            "comb_trigger_frame_size": "frame_size",
            "comb_trigger_hop_size": "hop_size",
            "comb_trigger_candidate_count": "candidate_count",
            "comb_trigger_harmonic_weight_count": "harmonic_weight_count",
        }

        for legacy_key, new_key in comb_aliases.items():
            if legacy_key in normalized:
                comb_config_data[new_key] = normalized.pop(legacy_key)

        if "expected_f0" not in normalized and "sr_augment_factor" in normalized:
            try:
                sr_factor = float(normalized["sr_augment_factor"])
            except (TypeError, ValueError):
                sr_factor = float("nan")
            if np.isfinite(sr_factor) and sr_factor > 0:
                normalized["expected_f0"] = 1000.0 / sr_factor

        known = {f.name for f in dataclasses.fields(PitchCompareConfig)}
        filtered = {k: v for k, v in normalized.items() if k in known}
        filtered["comb_trigger"] = HarmonicCombConfig(**comb_config_data)
        return PitchCompareConfig(**filtered)


def load_config(path: Path) -> PitchCompareConfig:
    data = json.loads(path.read_text())
    return PitchCompareConfig.from_dict(data)


def ensure_output_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
