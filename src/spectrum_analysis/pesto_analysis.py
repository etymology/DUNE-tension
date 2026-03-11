"""Helpers for estimating pitch from audio with PESTO."""

from __future__ import annotations

import logging
from typing import Any, Optional, Tuple

import numpy as np

torch = None  # type: ignore
load_model = None  # type: ignore
_RUNTIME_DEPS_LOADED = False
_MODEL_CACHE: dict[tuple[str, float, int], Any] = {}
DEFAULT_PESTO_MODEL_NAME = "mir-1k_g7"
DEFAULT_PESTO_STEP_SIZE_MS = 5.0
LOGGER = logging.getLogger(__name__)


def _to_numpy(value: Any) -> np.ndarray:
    if torch is not None and isinstance(value, torch.Tensor):  # type: ignore[union-attr]
        return value.detach().cpu().numpy()
    return np.asarray(value)


def _resolve_step_size_ms(_sample_rate: int, _sample_count: int) -> float:
    return float(max(DEFAULT_PESTO_STEP_SIZE_MS, 1.0))


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

    if sample_rate <= 0:
        raise ValueError("sample_rate must be positive.")

    if not _ensure_runtime_dependencies():
        LOGGER.warning("pesto-pitch is unavailable; cannot estimate pitch.")
        return float("nan"), float("nan")

    audio_array = np.asarray(audio, dtype=np.float32).reshape(-1)
    if audio_array.size == 0:
        return float("nan"), float("nan")

    step_size_ms = _resolve_step_size_ms(int(sample_rate), int(audio_array.size))
    model = _load_pesto_model_cached(
        DEFAULT_PESTO_MODEL_NAME,
        step_size_ms,
        int(sample_rate),
    )
    if model is None:
        LOGGER.warning("pesto model loader is unavailable; cannot estimate pitch.")
        return float("nan"), float("nan")

    audio_tensor = torch.from_numpy(audio_array).to(dtype=torch.float32).unsqueeze(0)

    try:
        with torch.inference_mode():
            predictions, confidences, _ = model(
                audio_tensor,
                sr=int(sample_rate),
                convert_to_freq=True,
                return_activations=False,
            )
    except Exception as exc:  # pragma: no cover - environment-specific inference failure
        LOGGER.warning("PESTO pitch estimation failed: %s", exc)
        return float("nan"), float("nan")

    predicted_frequencies = _to_numpy(predictions).reshape(-1)
    confidence_values = _to_numpy(confidences).reshape(-1)

    valid = np.isfinite(predicted_frequencies) & (predicted_frequencies > 0.0)
    valid &= np.isfinite(confidence_values) & (confidence_values > 0.0)
    if not np.any(valid):
        return float("nan"), float("nan")

    if expected_frequency is not None:
        try:
            max_allowed = float(expected_frequency) * 1.5
        except (TypeError, ValueError):
            max_allowed = float("nan")
        if np.isfinite(max_allowed) and max_allowed > 0.0:
            expected_mask = valid & (predicted_frequencies <= max_allowed)
            if np.any(expected_mask):
                valid = expected_mask

    weighted_frequencies = predicted_frequencies[valid]
    weights = confidence_values[valid]
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        frequency = float(np.mean(weighted_frequencies))
    else:
        frequency = float(np.average(weighted_frequencies, weights=weights))

    confidence = float(np.mean(weights))
    return frequency, confidence
