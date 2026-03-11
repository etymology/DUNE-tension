"""Helpers for estimating pitch from audio with PESTO."""

from __future__ import annotations

from dataclasses import dataclass
import logging
from typing import Any, Optional, Tuple

import numpy as np

torch = None  # type: ignore
load_model = None  # type: ignore
_RUNTIME_DEPS_LOADED = False
_MODEL_CACHE: dict[tuple[str, float, int], Any] = {}
DEFAULT_PESTO_MODEL_NAME = "mir-1k_g7"
DEFAULT_PESTO_STEP_SIZE_MS = 5.0
DEFAULT_PESTO_IDEAL_PITCH_HZ = 600.0
LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class PestoAnalysisResult:
    """Pitch estimate and optional activation diagnostics for one audio buffer."""

    frequency: float
    confidence: float
    frame_times: np.ndarray
    predicted_frequencies: np.ndarray
    frame_confidences: np.ndarray
    activation_map: np.ndarray | None = None
    activation_freq_axis: np.ndarray | None = None


def _to_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):  # type: ignore[union-attr]
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _resolve_step_size_ms(_sample_rate: int, _sample_count: int) -> float:
    return float(max(DEFAULT_PESTO_STEP_SIZE_MS, 1.0))


def _empty_analysis_result() -> PestoAnalysisResult:
    empty = np.zeros(0, dtype=np.float32)
    return PestoAnalysisResult(
        frequency=float("nan"),
        confidence=float("nan"),
        frame_times=empty,
        predicted_frequencies=empty,
        frame_confidences=empty,
        activation_map=None,
        activation_freq_axis=None,
    )


def _activation_frequency_axis(model: Any, num_bins: int) -> np.ndarray:
    bins_per_semitone = max(int(getattr(model, "bins_per_semitone", 1)), 1)
    preprocessor = getattr(model, "preprocessor", None)
    hcqt_kwargs = getattr(preprocessor, "hcqt_kwargs", {}) if preprocessor is not None else {}
    fmin = float(hcqt_kwargs.get("fmin", 32.7))
    return (
        fmin
        * np.power(
            2.0,
            np.arange(num_bins, dtype=np.float32) / (12.0 * bins_per_semitone),
        )
    ).astype(np.float32, copy=False)


def _sr_augment_factor(expected_frequency: Optional[float]) -> float:
    if expected_frequency is None:
        return 1.0

    try:
        expected = float(expected_frequency)
    except (TypeError, ValueError):
        expected = float("nan")

    if not np.isfinite(expected) or expected <= 0.0:
        return 1.0

    return DEFAULT_PESTO_IDEAL_PITCH_HZ / expected


def _reverse_sr_augment(
    activation: np.ndarray,
    sr_augment_factor: float,
    bins_per_semitone: int,
) -> np.ndarray:
    """Shift PESTO activations back onto the original pitch axis."""

    num_bins = activation.shape[1]
    bin_shift = int(round(-np.log2(sr_augment_factor) * 12.0 * bins_per_semitone))
    if abs(bin_shift) >= num_bins:
        return np.zeros_like(activation)
    if bin_shift > 0:
        return np.pad(activation, ((0, 0), (bin_shift, 0)), mode="constant")[:, :num_bins]
    if bin_shift < 0:
        return np.pad(activation, ((0, 0), (0, -bin_shift)), mode="constant")[
            :,
            -bin_shift:,
        ]
    return activation


def _ensure_runtime_dependencies() -> bool:
    global torch, load_model, _RUNTIME_DEPS_LOADED

    if torch is not None and load_model is not None:
        return True
    if _RUNTIME_DEPS_LOADED:
        return False

    try:  # pragma: no cover - dependency availability depends on environment
        import torch as torch_module  # type: ignore
        from pesto import load_model as pesto_load_model  # type: ignore
    except Exception:
        torch = None  # type: ignore
        load_model = None  # type: ignore
        _RUNTIME_DEPS_LOADED = True
        return False

    torch = torch_module  # type: ignore
    load_model = pesto_load_model  # type: ignore
    _RUNTIME_DEPS_LOADED = True
    return True


def _load_pesto_model_cached(model_name: str, step_size_ms: float, sample_rate: int) -> Any:
    if load_model is None:
        return None

    cache_key = (str(model_name), float(step_size_ms), int(sample_rate))
    model = _MODEL_CACHE.get(cache_key)
    if model is None:
        model = load_model(
            model_name,
            step_size=float(step_size_ms),
            sampling_rate=int(sample_rate),
            streaming=False,
            max_batch_size=1,
        )
        _MODEL_CACHE[cache_key] = model
    return model


def estimate_pitch_from_audio(
    audio: np.ndarray,
    sample_rate: int,
    expected_frequency: Optional[float] = None,
) -> Tuple[float, float]:
    """Estimate pitch for ``audio`` using PESTO.

    Returns ``(nan, nan)`` when inference dependencies are unavailable or the
    audio does not yield any voiced frames.
    """

    result = analyze_audio_with_pesto(
        audio,
        sample_rate,
        expected_frequency=expected_frequency,
        include_activations=False,
    )
    return result.frequency, result.confidence


def analyze_audio_with_pesto(
    audio: np.ndarray,
    sample_rate: int,
    expected_frequency: Optional[float] = None,
    *,
    include_activations: bool = False,
) -> PestoAnalysisResult:
    """Return pitch estimates and optional activation diagnostics for ``audio``."""

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    if not _ensure_runtime_dependencies():
        LOGGER.warning("pesto-pitch is unavailable; cannot estimate pitch.")
        return _empty_analysis_result()

    audio_array = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio_array.size == 0:
        return _empty_analysis_result()

    sr_augment_factor = _sr_augment_factor(expected_frequency)
    augmented_sample_rate = int(round(float(sample_rate) * sr_augment_factor))
    step_size_ms = _resolve_step_size_ms(augmented_sample_rate, int(audio_array.size))
    model = _load_pesto_model_cached(
        DEFAULT_PESTO_MODEL_NAME,
        step_size_ms,
        augmented_sample_rate,
    )
    if model is None:
        LOGGER.warning("pesto model loader is unavailable; cannot estimate pitch.")
        return _empty_analysis_result()

    audio_tensor = torch.from_numpy(audio_array).to(dtype=torch.float32).unsqueeze(0)

    try:
        with torch.inference_mode():
            outputs = model(
                audio_tensor,
                sr=augmented_sample_rate,
                convert_to_freq=True,
                return_activations=bool(include_activations),
            )
    except Exception as exc:  # pragma: no cover - environment-specific inference failure
        LOGGER.warning("PESTO pitch estimation failed: %s", exc)
        return _empty_analysis_result()

    if len(outputs) < 2:
        return _empty_analysis_result()

    predictions = outputs[0]
    confidences = outputs[1]
    activations = outputs[3] if include_activations and len(outputs) >= 4 else None

    predicted_frequencies = _to_numpy(predictions).reshape(-1).astype(np.float32, copy=False)
    confidence_values = _to_numpy(confidences).reshape(-1).astype(np.float32, copy=False)
    frame_times = (
        np.arange(predicted_frequencies.size, dtype=np.float32) * (step_size_ms / 1000.0)
    )
    if sr_augment_factor != 1.0:
        predicted_frequencies = predicted_frequencies / float(sr_augment_factor)
        frame_times = frame_times * float(sr_augment_factor)

    valid = np.isfinite(predicted_frequencies) & (predicted_frequencies > 0.0)
    valid &= np.isfinite(confidence_values) & (confidence_values > 0.0)

    if expected_frequency is not None:
        try:
            max_allowed = float(expected_frequency) * 1.5
        except (TypeError, ValueError):
            max_allowed = float("nan")
        if np.isfinite(max_allowed) and max_allowed > 0.0:
            expected_mask = valid & (predicted_frequencies <= max_allowed)
            if np.any(expected_mask):
                valid = expected_mask

    if np.any(valid):
        weighted_frequencies = predicted_frequencies[valid]
        weights = confidence_values[valid]
        weight_sum = float(np.sum(weights))
        if weight_sum <= 0.0:
            frequency = float(np.mean(weighted_frequencies))
        else:
            frequency = float(np.average(weighted_frequencies, weights=weights))
        confidence = float(np.mean(weights))
    else:
        frequency = float("nan")
        confidence = float("nan")

    activation_map: np.ndarray | None = None
    activation_freq_axis: np.ndarray | None = None
    if activations is not None:
        activation_np = _to_numpy(activations)
        if activation_np.ndim == 3 and activation_np.shape[0] == 1:
            activation_np = activation_np[0]
        if activation_np.ndim == 2:
            bins_per_semitone = max(int(getattr(model, "bins_per_semitone", 1)), 1)
            if sr_augment_factor != 1.0:
                activation_np = _reverse_sr_augment(
                    activation_np,
                    sr_augment_factor,
                    bins_per_semitone,
                )
            activation_map = activation_np.T.astype(np.float32, copy=False)
            activation_freq_axis = _activation_frequency_axis(
                model,
                activation_map.shape[0],
            )

    return PestoAnalysisResult(
        frequency=frequency,
        confidence=confidence,
        frame_times=frame_times,
        predicted_frequencies=predicted_frequencies,
        frame_confidences=confidence_values,
        activation_map=activation_map,
        activation_freq_axis=activation_freq_axis,
    )
