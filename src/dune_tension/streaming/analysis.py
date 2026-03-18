from __future__ import annotations

from dataclasses import dataclass, field
import itertools
import logging
import math
import queue
import threading
import uuid
from typing import Any, Callable

import numpy as np

from dune_tension.streaming.models import (
    PitchResult,
    StreamingFrame,
    StreamingSegment,
    VoicedWindow,
)
from dune_tension.streaming.pose import interpolate_segment_pose
from dune_tension.streaming.runtime import TimedAudioChunk
from spectrum_analysis.comb_trigger import harmonic_comb_response
from spectrum_analysis.pesto_analysis import analyze_audio_with_pesto

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class StreamingAnalysisConfig:
    sample_rate: int
    frame_size: int = 2048
    hop_size: int = 1024
    candidate_count: int = 48
    harmonic_weight_count: int = 8
    min_harmonics: int = 3
    min_frequency_hz: float = 40.0
    max_frequency_hz: float = 1800.0
    comb_threshold: float = 0.08
    flatness_threshold: float = 0.65
    min_rms: float = 0.001
    expected_frequency_ratio: float = 0.08
    voiced_window_min_frames: int = 2
    voiced_window_gap_frames: int = 1
    pitch_queue_size: int = 64


@dataclass(frozen=True)
class VoicedAudioWindow:
    window: VoicedWindow
    audio: np.ndarray
    expected_frequency_hz: float | None
    source_mode: str
    max_comb_score: float = 0.0


@dataclass(frozen=True)
class CompletedPitchJob:
    job: VoicedAudioWindow
    pitch_result: PitchResult


class FastFrameAnalyzer:
    """Comb-based online frame analyzer shared by live and replay paths."""

    def __init__(self, config: StreamingAnalysisConfig) -> None:
        self.config = config
        self.window = np.hanning(self.config.frame_size).astype(np.float32)
        self.freq_bins = np.fft.rfftfreq(
            self.config.frame_size,
            d=1.0 / float(self.config.sample_rate),
        )
        self.weights = 1.0 / np.arange(
            1,
            max(1, int(self.config.harmonic_weight_count)) + 1,
            dtype=np.float64,
        )
        self._default_candidates = np.geomspace(
            float(self.config.min_frequency_hz),
            float(self.config.max_frequency_hz),
            num=max(1, int(self.config.candidate_count)),
        )

    def _candidate_band(self, expected_frequency_hz: float | None) -> np.ndarray:
        if expected_frequency_hz is None or not np.isfinite(expected_frequency_hz):
            return self._default_candidates
        center = float(expected_frequency_hz)
        half_width = max(center * float(self.config.expected_frequency_ratio), 5.0)
        f_min = max(float(self.config.min_frequency_hz), center - half_width)
        f_max = min(float(self.config.max_frequency_hz), center + half_width)
        if f_max <= f_min:
            return np.array([center], dtype=np.float64)
        return np.geomspace(
            f_min,
            f_max,
            num=max(6, int(self.config.candidate_count // 3)),
        )

    def analyze_chunk(
        self,
        chunk: TimedAudioChunk,
        *,
        segment: StreamingSegment,
        audio_chunk_ref: str | None = None,
        expected_frequency_hz: float | None = None,
    ) -> list[StreamingFrame]:
        audio = np.asarray(chunk.audio, dtype=np.float32).reshape(-1)
        if audio.size < self.config.frame_size:
            return []

        frames: list[StreamingFrame] = []
        candidates = self._candidate_band(expected_frequency_hz)
        expected_candidates = (
            self._candidate_band(expected_frequency_hz)
            if expected_frequency_hz is not None
            else None
        )
        for frame_index, frame_start in enumerate(
            range(0, audio.size - self.config.frame_size + 1, self.config.hop_size)
        ):
            frame_audio = audio[frame_start : frame_start + self.config.frame_size]
            frame_mid_time = chunk.start_time + (
                (frame_start + (self.config.frame_size / 2)) / float(self.config.sample_rate)
            )
            pose, in_cruise = interpolate_segment_pose(segment, frame_mid_time)
            comb_score, spectral_flatness, harmonic_valid = harmonic_comb_response(
                frame_audio,
                self.config.sample_rate,
                self.window,
                self.freq_bins,
                candidates,
                self.weights,
                int(self.config.min_harmonics),
            )
            expected_band_score = None
            if expected_candidates is not None:
                expected_band_score, _, _ = harmonic_comb_response(
                    frame_audio,
                    self.config.sample_rate,
                    self.window,
                    self.freq_bins,
                    expected_candidates,
                    self.weights,
                    int(self.config.min_harmonics),
                )

            rms = float(np.sqrt(np.mean(np.square(frame_audio, dtype=np.float64))))
            voiced_gate_pass = bool(
                in_cruise
                and harmonic_valid
                and comb_score >= float(self.config.comb_threshold)
                and spectral_flatness <= float(self.config.flatness_threshold)
                and rms >= float(self.config.min_rms)
            )
            frames.append(
                StreamingFrame(
                    frame_id=f"{segment.segment_id}:{frame_index}",
                    segment_id=segment.segment_id,
                    timestamp=frame_mid_time,
                    pose=pose,
                    rms=rms,
                    comb_score=float(comb_score),
                    spectral_flatness=float(spectral_flatness),
                    harmonic_valid=bool(harmonic_valid),
                    expected_band_score=(
                        None if expected_band_score is None else float(expected_band_score)
                    ),
                    voiced_gate_pass=voiced_gate_pass,
                    audio_chunk_ref=audio_chunk_ref,
                )
            )
        return frames

    def build_voiced_windows(
        self,
        chunk: TimedAudioChunk,
        *,
        frames: list[StreamingFrame],
        segment_id: str | None,
        audio_chunk_ref: str | None,
        wire_hint: int | None = None,
        expected_frequency_hz: float | None = None,
        source_mode: str = "sweep",
    ) -> list[VoicedAudioWindow]:
        audio = np.asarray(chunk.audio, dtype=np.float32).reshape(-1)
        windows: list[VoicedAudioWindow] = []
        if not frames:
            return windows

        gap_limit = max(0, int(self.config.voiced_window_gap_frames))
        min_frames = max(1, int(self.config.voiced_window_min_frames))
        start_index: int | None = None
        last_voiced_index: int | None = None

        def finalize(end_index: int) -> None:
            nonlocal start_index, last_voiced_index
            if start_index is None or last_voiced_index is None:
                return
            voiced_count = last_voiced_index - start_index + 1
            if voiced_count < min_frames:
                start_index = None
                last_voiced_index = None
                return
            sample_start = start_index * self.config.hop_size
            sample_end = min(
                audio.size,
                (last_voiced_index * self.config.hop_size) + self.config.frame_size,
            )
            window_audio = audio[sample_start:sample_end].copy()
            frame_slice = frames[start_index : last_voiced_index + 1]
            pose_center = frame_slice[len(frame_slice) // 2].pose
            window_record = VoicedWindow(
                window_id=uuid.uuid4().hex,
                segment_id=segment_id,
                start_time=frame_slice[0].timestamp,
                end_time=frame_slice[-1].timestamp,
                pose_center=pose_center,
                wire_hint=wire_hint,
                audio_chunk_refs=(
                    [audio_chunk_ref] if audio_chunk_ref is not None else []
                ),
            )
            windows.append(
                VoicedAudioWindow(
                    window=window_record,
                    audio=window_audio,
                    expected_frequency_hz=expected_frequency_hz,
                    source_mode=source_mode,
                    max_comb_score=max(float(frame.comb_score) for frame in frame_slice),
                )
            )
            start_index = None
            last_voiced_index = None

        for index, frame in enumerate(frames):
            if frame.voiced_gate_pass:
                if start_index is None:
                    start_index = index
                last_voiced_index = index
                continue
            if (
                start_index is not None
                and last_voiced_index is not None
                and (index - last_voiced_index) > gap_limit
            ):
                finalize(index - 1)

        finalize(len(frames) - 1)
        return windows


class AsyncPitchWorker:
    """Background PESTO worker that prioritizes rescue windows over sweep windows."""

    def __init__(
        self,
        *,
        sample_rate: int,
        analyze_func: Callable[..., Any] | None = None,
        max_queue_size: int = 64,
        logger: logging.Logger | None = None,
    ) -> None:
        self.sample_rate = int(sample_rate)
        self._analyze_func = analyze_func or analyze_audio_with_pesto
        self._logger = logger or LOGGER
        self._queue: "queue.PriorityQueue[tuple[int, int, VoicedAudioWindow | None]]" = (
            queue.PriorityQueue(maxsize=max(1, int(max_queue_size)))
        )
        self._results: "queue.Queue[CompletedPitchJob]" = queue.Queue()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._sequence = itertools.count()
        self._started = False

    def start(self) -> None:
        if self._started:
            return
        self._started = True
        self._thread.start()

    def _run(self) -> None:
        while not self._stop_event.is_set() or not self._queue.empty():
            try:
                _priority, _sequence, job = self._queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                if job is None:
                    continue
                analysis = self._analyze_func(
                    job.audio,
                    self.sample_rate,
                    expected_frequency=job.expected_frequency_hz,
                    include_activations=False,
                )
                pitch_result = PitchResult(
                    pitch_result_id=uuid.uuid4().hex,
                    window_id=job.window.window_id,
                    frequency_hz=float(getattr(analysis, "frequency", float("nan"))),
                    confidence=float(getattr(analysis, "confidence", float("nan"))),
                    expected_frequency_hz=job.expected_frequency_hz,
                    frame_count=int(job.audio.size),
                    source_mode=str(job.source_mode),
                )
                self._results.put(CompletedPitchJob(job=job, pitch_result=pitch_result))
            except Exception as exc:
                self._logger.warning("Async pitch analysis failed: %s", exc)
            finally:
                self._queue.task_done()

    def submit(self, job: VoicedAudioWindow) -> bool:
        self.start()
        priority = 0 if str(job.source_mode) == "rescue" else 1
        item = (priority, next(self._sequence), job)
        try:
            self._queue.put_nowait(item)
            return True
        except queue.Full:
            if priority > 0:
                self._logger.debug("Dropping sweep pitch job because queue is full.")
                return False
            self._queue.put(item, timeout=1.0)
            return True

    def join(self) -> None:
        self._queue.join()

    def drain_results(self) -> list[CompletedPitchJob]:
        completed: list[CompletedPitchJob] = []
        while True:
            try:
                completed.append(self._results.get_nowait())
            except queue.Empty:
                break
        return completed

    def stop(self) -> None:
        self._stop_event.set()
        self.join()
        self._queue.put_nowait((0, next(self._sequence), None))
        self._thread.join(timeout=1.0)
