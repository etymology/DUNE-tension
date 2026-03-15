from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
import os
import sqlite3
import sys
from types import SimpleNamespace
from typing import Any, Callable, Iterator, Mapping

try:  # pragma: no cover - fallback for legacy test stubs
    import dune_tension.data_cache as data_cache
except Exception:  # pragma: no cover
    try:
        import data_cache  # type: ignore
    except Exception:  # pragma: no cover
        data_cache = SimpleNamespace(
            append_dataframe_row=lambda _path, _row: None,
            append_results_row=lambda _path, _row: None,
            append_dataframe_rows=lambda _path, _rows, **_kwargs: None,
            append_results_rows=lambda _path, _rows, **_kwargs: None,
            connect_write_database=lambda _path: sqlite3.connect(":memory:"),
            ensure_tables=lambda _conn: None,
        )

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


def _build_wire_position_provider() -> Any | None:
    try:
        from dune_tension.tensiometer_functions import WirePositionProvider
    except Exception:  # pragma: no cover - fallback for legacy test stubs
        try:
            from tensiometer_functions import WirePositionProvider  # type: ignore
        except Exception:
            return None
    return WirePositionProvider()


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
        get_cached_xy = getattr(plc, "get_cached_xy", get_xy)
        goto_xy = getattr(plc, "goto_xy", None)

        spoof_get_xy = getattr(plc, "spoof_get_xy", lambda: (0.0, 0.0))
        spoof_goto_xy = getattr(plc, "spoof_goto_xy", lambda *_args, **_kwargs: True)

        try:
            web_ok = bool(is_web_server_active())
        except Exception:
            web_ok = False

        if (
            not spoof_movement
            and web_ok
            and get_cached_xy is not None
            and goto_xy is not None
        ):
            active_get_xy = get_cached_xy
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

    def __init__(self, data_path: str, sample_batch_size: int = 25) -> None:
        self.data_path = data_path
        self.sample_batch_size = max(1, int(sample_batch_size))
        self._conn: sqlite3.Connection | None = None
        self._scope_depth = 0
        self._schema_ready = False
        self._sample_buffer: list[dict[str, Any]] = []

    def _row_for(self, result: TensionResult) -> dict[str, Any]:
        return {col: getattr(result, col, None) for col in EXPECTED_COLUMNS}

    def _ensure_connection(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = data_cache.connect_write_database(self.data_path)
            self._schema_ready = False
        if not self._schema_ready:
            data_cache.ensure_tables(self._conn)
            self._schema_ready = True
        return self._conn

    @contextmanager
    def run_scope(self) -> Iterator["ResultRepository"]:
        self._scope_depth += 1
        if self._scope_depth == 1:
            self._ensure_connection()
        try:
            yield self
        finally:
            self._scope_depth -= 1
            if self._scope_depth == 0:
                self.close()

    def _flush_samples(self, *, commit: bool) -> None:
        if not self._sample_buffer:
            return

        rows = self._sample_buffer
        self._sample_buffer = []

        if self._scope_depth > 0:
            conn = self._ensure_connection()
            data_cache.append_results_rows(
                self.data_path,
                rows,
                conn=conn,
                ensure_schema=False,
                commit=False,
            )
            if commit:
                conn.commit()
            return

        data_cache.append_results_rows(self.data_path, rows)

    def flush(self) -> None:
        if self._scope_depth > 0:
            conn = self._ensure_connection()
            self._flush_samples(commit=False)
            conn.commit()
            return
        self._flush_samples(commit=True)

    def append_sample(self, result: TensionResult) -> None:
        row = self._row_for(result)
        if self._scope_depth > 0:
            self._sample_buffer.append(row)
            if len(self._sample_buffer) >= self.sample_batch_size:
                self._flush_samples(commit=False)
            return
        data_cache.append_results_row(self.data_path, row)

    def append_result(self, result: TensionResult) -> None:
        row = self._row_for(result)
        if self._scope_depth > 0:
            conn = self._ensure_connection()
            data_cache.append_dataframe_rows(
                self.data_path,
                [row],
                conn=conn,
                ensure_schema=False,
                commit=False,
            )
            conn.commit()
            return
        data_cache.append_dataframe_row(self.data_path, row)

    def close(self) -> None:
        try:
            if self._conn is not None:
                self._flush_samples(commit=False)
                self._conn.commit()
        finally:
            if self._conn is not None:
                self._conn.close()
            self._conn = None
            self._schema_ready = False
            self._sample_buffer = []


@dataclass(frozen=True)
class RuntimeOptions:
    spoof_audio: bool = False
    spoof_movement: bool = False
    spoof_servo: bool = False
    spoof_valve: bool = False


@dataclass(frozen=True)
class RuntimeBundle:
    motion: MotionService
    audio: AudioCaptureService
    servo_controller: Any
    valve_controller: Any | None
    strum: Callable[[], None]
    repository_factory: Callable[[str], ResultRepository]
    wire_position_provider: Any | None = None

    def build_repository(self, data_path: str) -> ResultRepository:
        return self.repository_factory(data_path)


def resolve_runtime_options(
    environ: Mapping[str, str] | None = None,
) -> RuntimeOptions:
    active_environ = os.environ if environ is None else environ
    return RuntimeOptions(
        spoof_audio=bool(active_environ.get("SPOOF_AUDIO")),
        spoof_movement=bool(active_environ.get("SPOOF_PLC")),
        spoof_servo=bool(active_environ.get("SPOOF_SERVO")),
        spoof_valve=bool(active_environ.get("SPOOF_VALVE")),
    )


def _create_servo_controller(spoof_servo: bool) -> Any:
    try:
        from dune_tension.maestro import Controller, DummyController, ServoController
    except Exception:  # pragma: no cover - fallback for legacy test stubs
        try:
            from maestro import Controller, DummyController, ServoController  # type: ignore
        except Exception:
            return SimpleNamespace(
                focus_position=4000,
                focus_target=lambda _target: None,
                nudge_focus=lambda _delta: None,
                on_focus_command=None,
            )

    if spoof_servo:
        return ServoController(servo=DummyController())
    return ServoController(Controller())


def _create_valve_controller(spoof_valve: bool) -> Any | None:
    if spoof_valve:
        return None

    try:
        from dune_tension.hardware.valve_trigger import (
            DeviceNotFoundError,
            ValveController,
        )
    except Exception:  # pragma: no cover - fallback for legacy test stubs
        try:
            from hardware.valve_trigger import (  # type: ignore
                DeviceNotFoundError,
                ValveController,
            )
        except Exception:
            return None

    try:
        return ValveController()
    except (DeviceNotFoundError, RuntimeError) as exc:
        LOGGER.warning("Unable to initialise valve controller: %s", exc)
        return None


def _make_strum_callback(controller: Any | None) -> Callable[[], None]:
    if controller is None:
        return lambda: None

    def _strum() -> None:
        try:
            controller.pulse(0.002)
        except Exception as exc:
            LOGGER.warning("Valve pulse failed: %s", exc)

    return _strum


def build_runtime_bundle(options: RuntimeOptions | None = None) -> RuntimeBundle:
    active_options = options or resolve_runtime_options()
    valve_controller = _create_valve_controller(active_options.spoof_valve)
    return RuntimeBundle(
        motion=MotionService.build(spoof_movement=active_options.spoof_movement),
        audio=AudioCaptureService.build(spoof=active_options.spoof_audio),
        servo_controller=_create_servo_controller(active_options.spoof_servo),
        valve_controller=valve_controller,
        strum=_make_strum_callback(valve_controller),
        repository_factory=lambda data_path: ResultRepository(data_path),
        wire_position_provider=_build_wire_position_provider(),
    )
