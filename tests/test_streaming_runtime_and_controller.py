from __future__ import annotations

import sys
from pathlib import Path
import tempfile
import time
import types

import numpy as np

# Ensure src is on path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dune_tension.streaming.analysis import (  # noqa: E402
    AsyncPitchWorker,
    FastFrameAnalyzer,
    StreamingAnalysisConfig,
)
from dune_tension.streaming.controller import (  # noqa: E402
    StreamingControllerConfig,
    StreamingMeasurementController,
    SweepCorridor,
)
from dune_tension.streaming.pose import build_measurement_pose  # noqa: E402
from dune_tension.streaming.runtime import (  # noqa: E402
    AudioStreamService,
    MeasurementRuntime,
    TimedAudioChunk,
)
from dune_tension.streaming.storage import StreamingSessionRepository  # noqa: E402
from dune_tension.streaming.wire_positions import StreamingWirePositionProvider  # noqa: E402
from dune_tension.streaming.models import StreamingSegment, VoicedWindow  # noqa: E402
from dune_tension.streaming.focus_plane import FocusPlaneModel  # noqa: E402


def _harmonic_audio(sample_rate: int, duration_s: float) -> np.ndarray:
    t = np.arange(int(sample_rate * duration_s), dtype=np.float64) / sample_rate
    return (
        0.8 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.3 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.2 * np.sin(2.0 * np.pi * 660.0 * t)
    ).astype(np.float32)


def test_fast_frame_analyzer_detects_voiced_windows() -> None:
    sample_rate = 16000
    audio = _harmonic_audio(sample_rate, 0.6)
    analyzer = FastFrameAnalyzer(StreamingAnalysisConfig(sample_rate=sample_rate))
    chunk = TimedAudioChunk(
        audio=audio,
        start_time=0.0,
        end_time=float(audio.size / sample_rate),
        sample_rate=sample_rate,
    )
    pose = build_measurement_pose(
        x_true=1.0,
        y_true=2.0,
        focus=4000.0,
        focus_reference=4000.0,
        side="A",
    )
    segment = StreamingSegment(
        segment_id="segment",
        mode="sweep",
        pose0=pose,
        pose1=pose,
        speed_mm_s=0.0,
        planned_start_time=0.0,
        planned_end_time=chunk.end_time,
        cruise_start_time=0.0,
        cruise_end_time=chunk.end_time,
        segment_status="completed",
    )

    frames = analyzer.analyze_chunk(chunk, segment=segment)
    windows = analyzer.build_voiced_windows(
        chunk,
        frames=frames,
        segment_id=segment.segment_id,
        audio_chunk_ref="chunk-1",
        source_mode="sweep",
    )

    assert any(frame.voiced_gate_pass for frame in frames)
    assert len(windows) >= 1
    assert windows[0].max_comb_score > 0.0


def test_async_pitch_worker_uses_injected_pitch_analyzer() -> None:
    worker = AsyncPitchWorker(
        sample_rate=16000,
        analyze_func=lambda *_args, **_kwargs: types.SimpleNamespace(
            frequency=221.0,
            confidence=0.92,
        ),
    )
    window = VoicedWindow(
        window_id="window-1",
        segment_id="segment-1",
        start_time=0.0,
        end_time=0.1,
        pose_center=build_measurement_pose(
            x_true=1.0,
            y_true=2.0,
            focus=4000.0,
            focus_reference=4000.0,
            side="A",
        ),
    )
    from dune_tension.streaming.analysis import VoicedAudioWindow

    worker.submit(
        VoicedAudioWindow(
            window=window,
            audio=_harmonic_audio(16000, 0.2),
            expected_frequency_hz=220.0,
            source_mode="rescue",
            max_comb_score=0.7,
        )
    )
    worker.join()
    completed = worker.drain_results()

    assert len(completed) == 1
    assert completed[0].pitch_result.frequency_hz == 221.0
    assert completed[0].pitch_result.confidence == 0.92


class _FakeMotion:
    def __init__(self) -> None:
        self.moves = []

    def goto_xy(self, x: float, y: float, **kwargs) -> bool:
        self.moves.append((x, y, kwargs))
        time.sleep(0.05)
        return True

    def set_speed(self, **_kwargs):
        return True


class _FakeServo:
    def __init__(self) -> None:
        self.focus_position = 4000

    def focus_target(self, target: int) -> None:
        self.focus_position = int(target)


class _FakeResultRepository:
    def __init__(self, sink: list) -> None:
        self._sink = sink

    def append_result(self, result) -> None:
        self._sink.append(result)

    def close(self) -> None:
        return None


class _FakeWireProvider:
    def __init__(self) -> None:
        self.positions = {1: (10.0, 20.0)}

    def invalidate(self) -> None:
        return None

    def get_xy(self, config, wire_number: int):
        return self.positions.get(int(wire_number))


class _FakeAudioSource:
    def __init__(self, chunks: list[np.ndarray], sample_rate: int) -> None:
        self._chunks = list(chunks)
        self._sample_rate = sample_rate
        self._stopped = False

    def start(self) -> None:
        return None

    def read(self) -> np.ndarray:
        if self._stopped:
            return np.array([], dtype=np.float32)
        if self._chunks:
            chunk = self._chunks.pop(0)
            time.sleep(chunk.size / self._sample_rate)
            return chunk
        time.sleep(0.01)
        return np.array([], dtype=np.float32)

    def stop(self) -> None:
        self._stopped = True


def test_streaming_controller_run_sweep_accepts_candidate(tmp_path) -> None:
    sample_rate = 16000
    recorded_results: list = []
    audio = _harmonic_audio(sample_rate, 0.4)
    audio_chunks = [audio[index : index + 2048] for index in range(0, audio.size, 2048)]
    runtime = MeasurementRuntime(
        motion=_FakeMotion(),
        servo_controller=_FakeServo(),
        valve_controller=None,
        strum=lambda: None,
        result_repository_factory=lambda _path: _FakeResultRepository(recorded_results),
        streaming_repository_factory=lambda session_id=None: StreamingSessionRepository(
            root_dir=tmp_path,
            session_id=session_id or "session-1",
        ),
        audio_stream_factory=lambda sr, hop: AudioStreamService(
            sample_rate=sr,
            hop_size=hop,
            source_factory=lambda: _FakeAudioSource(audio_chunks, sr),
        ),
        wire_positions=StreamingWirePositionProvider(provider=_FakeWireProvider()),
        focus_plane=FocusPlaneModel(c=4000.0),
        clock=time.monotonic,
        logger=types.SimpleNamespace(warning=lambda *args, **kwargs: None),
    )
    controller = StreamingMeasurementController(
        runtime=runtime,
        config=StreamingControllerConfig(
            apa_name="APA",
            layer="X",
            side="A",
            sample_rate=sample_rate,
            cruise_margin_s=0.0,
            direct_accept_support=1,
            direct_accept_confidence=0.5,
            direct_accept_angle_score=0.5,
            direct_accept_focus_score=0.5,
        ),
    )
    controller.analysis.config = StreamingAnalysisConfig(
        sample_rate=sample_rate,
        voiced_window_min_frames=1,
    )
    controller.analysis = FastFrameAnalyzer(controller.analysis.config)
    controller.pitch_worker._analyze_func = lambda *_args, **_kwargs: types.SimpleNamespace(
        frequency=220.0,
        confidence=0.91,
    )

    summary = controller.run_sweep(
        [
            SweepCorridor(
                corridor_id="corridor-1",
                x0=9.5,
                y0=20.0,
                x1=10.5,
                y1=20.0,
                speed_mm_s=5.0,
            )
        ]
    )

    assert summary["accepted_wires"] == [1]
    assert len(recorded_results) == 1
    assert (tmp_path / summary["session_id"] / "streaming.db").exists()
