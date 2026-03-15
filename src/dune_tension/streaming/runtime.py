from __future__ import annotations

from dataclasses import dataclass
import logging
import queue
import threading
import time
from typing import Any, Callable

import numpy as np

from dune_tension.services import (
    ResultRepository,
    RuntimeBundle,
    build_runtime_bundle,
    resolve_runtime_options,
)
from dune_tension.streaming.focus_plane import FocusPlaneModel
from dune_tension.streaming.storage import StreamingSessionRepository
from dune_tension.streaming.wire_positions import StreamingWirePositionProvider
from spectrum_analysis.audio_sources import AudioSource, DemoSource, MicSource, sd

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TimedAudioChunk:
    audio: np.ndarray
    start_time: float
    end_time: float
    sample_rate: int


class AudioStreamService:
    """Background microphone reader that timestamps audio hops."""

    def __init__(
        self,
        *,
        sample_rate: int,
        hop_size: int,
        source_factory: Callable[[], AudioSource] | None = None,
        clock: Callable[[], float] | None = None,
        queue_size: int = 256,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self.hop_size = int(hop_size)
        self._clock = clock or time.monotonic
        self._source_factory = source_factory or self._default_source_factory
        self._queue: "queue.Queue[TimedAudioChunk]" = queue.Queue(maxsize=max(1, int(queue_size)))
        self._source: AudioSource | None = None
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def _default_source_factory(self) -> AudioSource:
        if sd is None:
            return DemoSource(self.sample_rate, self.hop_size)
        return MicSource(self.sample_rate, self.hop_size)

    def start(self) -> None:
        if self._thread is not None and self._thread.is_alive():
            return
        self._stop_event.clear()
        self._source = self._source_factory()
        self._source.start()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        assert self._source is not None
        while not self._stop_event.is_set():
            chunk = self._source.read()
            if chunk.size == 0:
                continue
            chunk_array = np.asarray(chunk, dtype=np.float32).reshape(-1)
            chunk_end = float(self._clock())
            chunk_start = chunk_end - (chunk_array.size / max(self.sample_rate, 1))
            timed = TimedAudioChunk(
                audio=chunk_array,
                start_time=chunk_start,
                end_time=chunk_end,
                sample_rate=self.sample_rate,
            )
            try:
                self._queue.put_nowait(timed)
            except queue.Full:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    pass
                try:
                    self._queue.put_nowait(timed)
                except queue.Full:
                    pass

    def drain_available(self) -> list[TimedAudioChunk]:
        chunks: list[TimedAudioChunk] = []
        while True:
            try:
                chunks.append(self._queue.get_nowait())
            except queue.Empty:
                break
        return chunks

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._source is not None:
            self._source.stop()
        self._thread = None
        self._source = None


@dataclass(frozen=True)
class MeasurementRuntime:
    motion: Any
    servo_controller: Any
    valve_controller: Any | None
    strum: Callable[[], None]
    result_repository_factory: Callable[[str], ResultRepository]
    streaming_repository_factory: Callable[[str | None], StreamingSessionRepository]
    audio_stream_factory: Callable[[int, int], AudioStreamService]
    wire_positions: StreamingWirePositionProvider
    focus_plane: FocusPlaneModel
    clock: Callable[[], float]
    logger: logging.Logger


def build_measurement_runtime(
    *,
    runtime_bundle: RuntimeBundle | None = None,
    audio_source_factory: Callable[[], AudioSource] | None = None,
    clock: Callable[[], float] | None = None,
) -> MeasurementRuntime:
    active_bundle = runtime_bundle or build_runtime_bundle(resolve_runtime_options())
    active_clock = clock or time.monotonic
    wire_positions = StreamingWirePositionProvider(active_bundle.wire_position_provider)
    return MeasurementRuntime(
        motion=active_bundle.motion,
        servo_controller=active_bundle.servo_controller,
        valve_controller=active_bundle.valve_controller,
        strum=active_bundle.strum,
        result_repository_factory=active_bundle.repository_factory,
        streaming_repository_factory=lambda session_id=None: StreamingSessionRepository(
            session_id=session_id
        ),
        audio_stream_factory=lambda sample_rate, hop_size: AudioStreamService(
            sample_rate=sample_rate,
            hop_size=hop_size,
            source_factory=audio_source_factory,
            clock=active_clock,
        ),
        wire_positions=wire_positions,
        focus_plane=FocusPlaneModel(),
        clock=active_clock,
        logger=LOGGER,
    )
