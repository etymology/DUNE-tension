from __future__ import annotations

from dataclasses import dataclass
import logging
import sys
from typing import Any, Callable

try:  # pragma: no cover - fallback for legacy test stubs
    from dune_tension.data_cache import append_dataframe_row, append_results_row
except Exception:  # pragma: no cover
    try:
        from data_cache import append_dataframe_row, append_results_row  # type: ignore
    except Exception:  # pragma: no cover
        append_dataframe_row = lambda _path, _row: None  # type: ignore
        append_results_row = lambda _path, _row: None  # type: ignore

try:  # pragma: no cover - fallback for legacy test stubs
    from dune_tension.results import EXPECTED_COLUMNS, TensionResult
except Exception:  # pragma: no cover
    from results import EXPECTED_COLUMNS, TensionResult  # type: ignore

LOGGER = logging.getLogger(__name__)


def _import_plc_module() -> Any:
    plc_stub = sys.modules.get("plc_io")
    if plc_stub is not None:
        return plc_stub
    try:
        import dune_tension.plc_io as plc
    except Exception:  # pragma: no cover - fallback for legacy test stubs
        import plc_io as plc  # type: ignore
    return plc


def _import_audio_module() -> Any:
    audio_stub = sys.modules.get("audioProcessing")
    if audio_stub is not None:
        return audio_stub
    try:
        import dune_tension.audioProcessing as audio
    except Exception:  # pragma: no cover - fallback for legacy test stubs
        import audioProcessing as audio  # type: ignore
    return audio


@dataclass(frozen=True)
class MotionService:
    get_xy: Callable[[], tuple[float, float]]
    goto_xy: Callable[..., bool]
    increment: Callable[[float, float], Any]
    reset_plc: Callable[..., Any]
    set_speed: Callable[..., Any]

    @classmethod
    def build(cls, spoof_movement: bool) -> "MotionService":
        plc = _import_plc_module()

        is_web_server_active = getattr(plc, "is_web_server_active", lambda: False)
        increment = getattr(plc, "increment", lambda *_args, **_kwargs: None)
        set_speed = getattr(plc, "set_speed", lambda *_args, **_kwargs: True)
        reset_plc = getattr(plc, "reset_plc", lambda *_args, **_kwargs: None)

        get_xy = getattr(plc, "get_xy", None)
        goto_xy = getattr(plc, "goto_xy", None)

        spoof_get_xy = getattr(plc, "spoof_get_xy", lambda: (0.0, 0.0))
        spoof_goto_xy = getattr(plc, "spoof_goto_xy", lambda *_args, **_kwargs: True)

        try:
            web_ok = bool(is_web_server_active())
        except Exception:
            web_ok = False

        if not spoof_movement and web_ok and get_xy is not None and goto_xy is not None:
            active_get_xy = get_xy
            active_goto_xy = goto_xy
        else:
            LOGGER.warning(
                "Web server is not active or spoof_movement enabled. Using dummy functions."
            )
            active_get_xy = spoof_get_xy
            active_goto_xy = spoof_goto_xy

        return cls(
            get_xy=active_get_xy,
            goto_xy=active_goto_xy,
            increment=increment,
            reset_plc=reset_plc,
            set_speed=set_speed,
        )


@dataclass(frozen=True)
class AudioCaptureService:
    samplerate: int
    noise_threshold: float
    record_audio: Callable[[float, int], tuple[Any, float]]

    @classmethod
    def build(cls, spoof: bool) -> "AudioCaptureService":
        audio = _import_audio_module()

        get_samplerate = getattr(audio, "get_samplerate", lambda: None)
        get_noise_threshold = getattr(audio, "get_noise_threshold", lambda: 0.0)
        record_audio_filtered = getattr(audio, "record_audio_filtered", None)
        spoof_audio_sample = getattr(audio, "spoof_audio_sample", None)

        samplerate = get_samplerate()
        noise_threshold = float(get_noise_threshold())

        if samplerate is None or spoof or record_audio_filtered is None:
            LOGGER.info("Using spoofed audio sample for testing.")
            samplerate = 44100
            if spoof_audio_sample is None:
                record_audio = lambda _duration, _sample_rate: ([], 0.0)
            else:
                record_audio = lambda _duration, _sample_rate: (
                    spoof_audio_sample("audio"),
                    0.0,
                )
        else:
            record_audio = lambda duration, sample_rate: record_audio_filtered(
                duration,
                sample_rate=sample_rate,
                normalize=True,
            )

        return cls(
            samplerate=int(samplerate),
            noise_threshold=noise_threshold,
            record_audio=record_audio,
        )


class ResultRepository:
    """Repository for persisting measured tension results."""

    def __init__(self, data_path: str) -> None:
        self.data_path = data_path

    def append_sample(self, result: TensionResult) -> None:
        row = {col: getattr(result, col, None) for col in EXPECTED_COLUMNS}
        append_results_row(self.data_path, row)

    def append_result(self, result: TensionResult) -> None:
        row = {col: getattr(result, col, None) for col in EXPECTED_COLUMNS}
        append_dataframe_row(self.data_path, row)
