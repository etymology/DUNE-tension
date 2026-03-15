from __future__ import annotations

from dataclasses import dataclass, field
import logging
import math
from pathlib import Path
import threading
import time
from typing import Callable, Iterable

import numpy as np

from dune_tension.geometry import length_lookup, zone_lookup
from dune_tension.results import TensionResult
from dune_tension.streaming.analysis import (
    AsyncPitchWorker,
    FastFrameAnalyzer,
    StreamingAnalysisConfig,
)
from dune_tension.streaming.evidence import PitchEvidenceField
from dune_tension.streaming.models import (
    FocusAnchor,
    PitchObservation,
    PulseEvent,
    RescueQueueItem,
    StreamingManifest,
    StreamingSegment,
    WireCandidate,
)
from dune_tension.streaming.pose import (
    build_measurement_pose,
    stage_x_for_laser_target,
)
from dune_tension.streaming.runtime import MeasurementRuntime
from dune_tension.streaming.storage import StreamingSessionRepository
from dune_tension.tensiometer_functions import make_config
from dune_tension.tension_calculation import wire_equation

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class SweepCorridor:
    corridor_id: str
    x0: float
    y0: float
    x1: float
    y1: float
    speed_mm_s: float
    focus_offset: float = 0.0


@dataclass
class StreamingControllerConfig:
    apa_name: str
    layer: str
    side: str
    flipped: bool = False
    sample_rate: int = 44100
    frame_size: int = 2048
    hop_size: int = 1024
    pulse_duration_s: float = 0.002
    pulse_interval_s: float = 0.75
    cruise_margin_s: float = 0.05
    direct_accept_confidence: float = 0.85
    direct_accept_support: int = 2
    direct_accept_angle_score: float = 0.5
    direct_accept_focus_score: float = 0.5
    candidate_radius_mm: float = 2.5
    focus_probe_step: float = 50.0
    rescue_position_step_mm: float = 0.5
    rescue_capture_seconds: float = 0.5
    default_focus: float = 4000.0
    data_path: str = field(init=False)

    def __post_init__(self) -> None:
        self.data_path = make_config(
            apa_name=self.apa_name,
            layer=self.layer,
            side=self.side,
            flipped=self.flipped,
        ).data_path

    def analysis_config(self) -> StreamingAnalysisConfig:
        return StreamingAnalysisConfig(
            sample_rate=self.sample_rate,
            frame_size=self.frame_size,
            hop_size=self.hop_size,
        )


class StreamingMeasurementController:
    """Headless controller for sweep and rescue streaming runs."""

    def __init__(
        self,
        *,
        runtime: MeasurementRuntime,
        config: StreamingControllerConfig,
        status_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self.runtime = runtime
        self.config = config
        self.status_callback = status_callback or (lambda _payload: None)
        self.analysis = FastFrameAnalyzer(config.analysis_config())
        self.pitch_worker = AsyncPitchWorker(
            sample_rate=self.config.sample_rate,
            max_queue_size=self.analysis.config.pitch_queue_size,
            logger=runtime.logger,
        )
        self.evidence_field = PitchEvidenceField()
        self._active_repo: StreamingSessionRepository | None = None
        self._active_session_id: str | None = None
        self._active_audio_stream = None

    def _notify(self, **payload: object) -> None:
        try:
            self.status_callback(payload)
        except Exception:
            return

    def _set_focus(self, target: float) -> None:
        try:
            self.runtime.servo_controller.focus_target(int(round(float(target))))
        except Exception as exc:
            self.runtime.logger.warning("Focus command failed: %s", exc)

    def _goto_stage_xy(self, x_stage: float, y_stage: float, *, speed_mm_s: float | None = None) -> bool:
        kwargs = {}
        if speed_mm_s is not None:
            kwargs["speed"] = float(speed_mm_s)
        try:
            return self.runtime.motion.goto_xy(float(x_stage), float(y_stage), **kwargs) is not False
        except TypeError:
            return self.runtime.motion.goto_xy(float(x_stage), float(y_stage)) is not False

    def _pulse_scheduler(
        self,
        *,
        start_time: float,
        end_time: float,
        stop_event: threading.Event,
        repo: StreamingSessionRepository,
        segment_id: str,
    ) -> None:
        next_pulse = start_time
        index = 0
        while not stop_event.is_set():
            now = self.runtime.clock()
            if now >= end_time:
                return
            if now < next_pulse:
                time.sleep(min(0.01, next_pulse - now))
                continue
            self.runtime.strum()
            repo.append_pulse_event(
                PulseEvent(
                    pulse_id=f"{segment_id}:pulse:{index}",
                    segment_id=segment_id,
                    timestamp=now,
                    duration_s=self.config.pulse_duration_s,
                )
            )
            index += 1
            next_pulse = now + float(self.config.pulse_interval_s)

    def _process_pitch_results(self, completed_jobs) -> None:
        if self._active_repo is None:
            return
        for completed in completed_jobs:
            self._active_repo.append_pitch_result(completed.pitch_result)
            if not np.isfinite(completed.pitch_result.frequency_hz):
                continue
            if not np.isfinite(completed.pitch_result.confidence):
                continue
            observation = PitchObservation(
                observation_id=completed.pitch_result.pitch_result_id,
                x_true=completed.job.window.pose_center.x_laser,
                y_true=completed.job.window.pose_center.y_true,
                frequency_hz=completed.pitch_result.frequency_hz,
                confidence=completed.pitch_result.confidence,
                comb_score=float(completed.job.max_comb_score),
                focus_delta=completed.job.window.pose_center.delta_focus,
                source_window_id=completed.job.window.window_id,
                source_sweep_id=self._active_session_id,
            )
            pitch_bin = self.evidence_field.observe(observation)
            self._active_repo.upsert_pitch_bin(pitch_bin)

    def _corridor_segment(
        self,
        *,
        corridor: SweepCorridor,
        segment_index: int,
    ) -> tuple[StreamingSegment, float, float, float, float]:
        x_mid = (float(corridor.x0) + float(corridor.x1)) / 2.0
        y_mid = (float(corridor.y0) + float(corridor.y1)) / 2.0
        focus_reference = self.runtime.focus_plane.predict(x_mid, y_mid, clamp=False)
        if not np.isfinite(focus_reference):
            focus_reference = float(self.config.default_focus)
        focus = float(focus_reference) + float(corridor.focus_offset)
        x_stage0 = stage_x_for_laser_target(
            x_laser_target=float(corridor.x0),
            focus=focus,
            focus_reference=focus_reference,
            side=self.config.side,
        )
        x_stage1 = stage_x_for_laser_target(
            x_laser_target=float(corridor.x1),
            focus=focus,
            focus_reference=focus_reference,
            side=self.config.side,
        )
        pose0 = build_measurement_pose(
            x_true=x_stage0,
            y_true=float(corridor.y0),
            focus=focus,
            focus_reference=focus_reference,
            side=self.config.side,
        )
        pose1 = build_measurement_pose(
            x_true=x_stage1,
            y_true=float(corridor.y1),
            focus=focus,
            focus_reference=focus_reference,
            side=self.config.side,
        )
        distance_mm = math.hypot(x_stage1 - x_stage0, float(corridor.y1) - float(corridor.y0))
        duration_s = distance_mm / max(float(corridor.speed_mm_s), 1e-6)
        planned_start = self.runtime.clock()
        planned_end = planned_start + duration_s
        cruise_start = planned_start + float(self.config.cruise_margin_s)
        cruise_end = planned_end - float(self.config.cruise_margin_s)
        segment = StreamingSegment(
            segment_id=f"{corridor.corridor_id}:{segment_index}",
            mode="sweep",
            pose0=pose0,
            pose1=pose1,
            speed_mm_s=float(corridor.speed_mm_s),
            planned_start_time=planned_start,
            planned_end_time=planned_end,
            cruise_start_time=cruise_start,
            cruise_end_time=cruise_end,
            segment_status="planned",
        )
        return segment, x_stage0, float(corridor.y0), x_stage1, float(corridor.y1)

    def _analyze_chunks_for_segment(
        self,
        *,
        repo: StreamingSessionRepository,
        segment: StreamingSegment,
        chunks,
        expected_frequency_hz: float | None,
        source_mode: str,
        wire_hint: int | None = None,
    ) -> None:
        for chunk_index, chunk in enumerate(chunks):
            ref = repo.append_audio_chunk(
                audio=chunk.audio,
                sample_rate=chunk.sample_rate,
                start_time=chunk.start_time,
                end_time=chunk.end_time,
                segment_id=segment.segment_id,
                chunk_id=f"{segment.segment_id}:chunk:{chunk_index}",
            )
            frames = self.analysis.analyze_chunk(
                chunk,
                segment=segment,
                audio_chunk_ref=ref.chunk_id,
                expected_frequency_hz=expected_frequency_hz,
            )
            for frame in frames:
                repo.append_frame(frame)
            windows = self.analysis.build_voiced_windows(
                chunk,
                frames=frames,
                segment_id=segment.segment_id,
                audio_chunk_ref=ref.chunk_id,
                wire_hint=wire_hint,
                expected_frequency_hz=expected_frequency_hz,
                source_mode=source_mode,
            )
            for window in windows:
                repo.append_voiced_window(window.window)
                self.pitch_worker.submit(window)

    def _coherence_score(self, points, direction, competitors) -> float:
        if len(points) < 2:
            return 0.5

        def score_for(vector) -> float:
            vx, vy = vector
            perp = np.array([-vy, vx], dtype=np.float64)
            point_matrix = np.asarray(points, dtype=np.float64)
            perp_projection = point_matrix @ perp
            return 1.0 / (1.0 + float(np.std(perp_projection)))

        active_score = score_for(direction)
        competitor_score = max((score_for(item) for item in competitors), default=0.0)
        if active_score <= 0.0 and competitor_score <= 0.0:
            return 0.0
        return active_score / (active_score + competitor_score)

    def _aggregate_candidates(self, *, source_mode: str) -> dict[int, WireCandidate]:
        candidates: dict[int, WireCandidate] = {}
        points_by_wire: dict[int, list[tuple[float, float]]] = {}
        for pitch_bin in self.evidence_field.snapshot():
            if not pitch_bin.hypotheses:
                continue
            dominant = pitch_bin.hypotheses[0]
            nearby = self.runtime.wire_positions.nearby_wires(
                apa_name=self.config.apa_name,
                layer=self.config.layer,
                side=self.config.side,
                flipped=self.config.flipped,
                x_laser=pitch_bin.x_bin,
                y_true=pitch_bin.y_bin,
                radius_mm=self.config.candidate_radius_mm,
            )
            for predicted in nearby:
                focus_score = 0.5
                if dominant.focus_response:
                    best_focus = max(
                        dominant.focus_response,
                        key=lambda item: float(item.combined_confidence),
                    )
                    focus_score = max(
                        0.0,
                        1.0 - (abs(float(best_focus.delta_focus)) / max(self.config.focus_probe_step, 1.0)),
                    )
                pose = build_measurement_pose(
                    x_true=pitch_bin.x_bin,
                    y_true=pitch_bin.y_bin,
                    focus=self.runtime.focus_plane.predict(
                        pitch_bin.x_bin,
                        pitch_bin.y_bin,
                        clamp=False,
                    ),
                    focus_reference=self.runtime.focus_plane.predict(
                        pitch_bin.x_bin,
                        pitch_bin.y_bin,
                        clamp=False,
                    ),
                    side=self.config.side,
                )
                candidate = candidates.get(predicted.wire_number)
                if candidate is None:
                    candidate = WireCandidate(
                        wire_number=int(predicted.wire_number),
                        source_mode=str(source_mode),
                        support_count=0,
                        best_pose=pose,
                        best_comb_score=float(dominant.max_comb_score),
                        stream_session_id=self._active_session_id,
                    )
                    candidates[predicted.wire_number] = candidate
                    points_by_wire[predicted.wire_number] = []
                candidate.support_count += int(dominant.support_count)
                candidate.pitch_estimates.append(float(dominant.weighted_pitch_hz))
                candidate.pitch_confidences.append(float(dominant.combined_confidence))
                candidate.focus_consistency_score = max(
                    float(candidate.focus_consistency_score),
                    float(focus_score),
                )
                if float(dominant.max_comb_score) >= float(candidate.best_comb_score):
                    candidate.best_comb_score = float(dominant.max_comb_score)
                    candidate.best_pose = pose
                points_by_wire[predicted.wire_number].append((pitch_bin.x_bin, pitch_bin.y_bin))

        direction = self.runtime.wire_positions.wire_direction(
            layer=self.config.layer,
            side=self.config.side,
            flipped=self.config.flipped,
        )
        competitors = self.runtime.wire_positions.competing_directions(
            layer=self.config.layer,
            side=self.config.side,
            flipped=self.config.flipped,
        )
        for wire_number, candidate in candidates.items():
            candidate.angle_coherence_score = self._coherence_score(
                points_by_wire.get(wire_number, []),
                direction,
                competitors,
            )
        return candidates

    def _persist_final_result(self, candidate: WireCandidate, measurement_mode: str) -> TensionResult:
        avg_frequency = float(np.average(candidate.pitch_estimates, weights=np.maximum(candidate.pitch_confidences, 1e-6)))
        avg_confidence = float(np.mean(candidate.pitch_confidences))
        result = TensionResult(
            apa_name=self.config.apa_name,
            layer=self.config.layer,
            side=self.config.side,
            wire_number=int(candidate.wire_number),
            frequency=avg_frequency,
            confidence=avg_confidence,
            x=float(candidate.best_pose.x_laser),
            y=float(candidate.best_pose.y_true),
            focus_position=int(round(candidate.best_pose.focus)),
            measurement_mode=measurement_mode,
            stream_session_id=self._active_session_id,
        )
        repository = self.runtime.result_repository_factory(self.config.data_path)
        repository.append_result(result)
        repository.close()
        return result

    def run_sweep(
        self,
        corridors: Iterable[SweepCorridor],
        *,
        session_id: str | None = None,
    ) -> dict[str, object]:
        corridor_list = list(corridors)
        repo = self.runtime.streaming_repository_factory(session_id)
        self._active_repo = repo
        self._active_session_id = repo.session_id
        self.evidence_field = PitchEvidenceField()
        manifest = StreamingManifest(
            session_id=repo.session_id,
            measurement_mode="stream_sweep",
            apa_name=self.config.apa_name,
            layer=self.config.layer,
            side=self.config.side,
            flipped=self.config.flipped,
            runtime_config={
                "sample_rate": self.config.sample_rate,
                "frame_size": self.config.frame_size,
                "hop_size": self.config.hop_size,
                "pulse_interval_s": self.config.pulse_interval_s,
            },
            focus_plane_coefficients=self.runtime.focus_plane.coefficients(),
            sweep_corridors=[
                {
                    "corridor_id": corridor.corridor_id,
                    "x0": corridor.x0,
                    "y0": corridor.y0,
                    "x1": corridor.x1,
                    "y1": corridor.y1,
                    "speed_mm_s": corridor.speed_mm_s,
                    "focus_offset": corridor.focus_offset,
                }
                for corridor in corridor_list
            ],
        )
        repo.write_manifest(manifest)

        audio_stream = self.runtime.audio_stream_factory(
            self.config.sample_rate,
            self.config.hop_size,
        )
        self._active_audio_stream = audio_stream
        audio_stream.start()
        accepted_results: list[TensionResult] = []
        queued_for_rescue: list[int] = []
        try:
            for index, corridor in enumerate(corridor_list):
                segment, x_stage0, y0, x_stage1, y1 = self._corridor_segment(
                    corridor=corridor,
                    segment_index=index,
                )
                self._notify(segment_id=segment.segment_id, corridor_id=corridor.corridor_id)
                self._set_focus(segment.pose0.focus)
                self._goto_stage_xy(x_stage0, y0)
                pulse_stop = threading.Event()
                pulse_thread = threading.Thread(
                    target=self._pulse_scheduler,
                    kwargs={
                        "start_time": segment.cruise_start_time,
                        "end_time": segment.cruise_end_time,
                        "stop_event": pulse_stop,
                        "repo": repo,
                        "segment_id": segment.segment_id,
                    },
                    daemon=True,
                )
                repo.append_segment(segment)
                pulse_thread.start()
                self.runtime.motion.set_speed(speed=float(corridor.speed_mm_s))
                actual_start = self.runtime.clock()
                self._goto_stage_xy(x_stage1, y1, speed_mm_s=corridor.speed_mm_s)
                actual_end = self.runtime.clock()
                pulse_stop.set()
                pulse_thread.join(timeout=1.0)
                updated_segment = StreamingSegment(
                    segment_id=segment.segment_id,
                    mode=segment.mode,
                    pose0=segment.pose0,
                    pose1=segment.pose1,
                    speed_mm_s=segment.speed_mm_s,
                    planned_start_time=actual_start,
                    planned_end_time=actual_end,
                    cruise_start_time=actual_start + float(self.config.cruise_margin_s),
                    cruise_end_time=max(
                        actual_start + float(self.config.cruise_margin_s),
                        actual_end - float(self.config.cruise_margin_s),
                    ),
                    segment_status="completed",
                )
                repo.append_segment(updated_segment)
                time.sleep(0.1)
                chunks = [
                    chunk
                    for chunk in audio_stream.drain_available()
                    if chunk.end_time >= actual_start and chunk.start_time <= actual_end
                ]
                self._analyze_chunks_for_segment(
                    repo=repo,
                    segment=updated_segment,
                    chunks=chunks,
                    expected_frequency_hz=None,
                    source_mode="sweep",
                )
                self.pitch_worker.join()
                self._process_pitch_results(self.pitch_worker.drain_results())

            candidates = self._aggregate_candidates(source_mode="sweep")
            for candidate in candidates.values():
                if (
                    candidate.support_count >= int(self.config.direct_accept_support)
                    and candidate.pitch_confidences
                    and float(np.mean(candidate.pitch_confidences))
                    >= float(self.config.direct_accept_confidence)
                    and float(candidate.angle_coherence_score)
                    >= float(self.config.direct_accept_angle_score)
                    and float(candidate.focus_consistency_score)
                    >= float(self.config.direct_accept_focus_score)
                ):
                    candidate.status = "accepted"
                    accepted_results.append(
                        self._persist_final_result(candidate, "stream_sweep")
                    )
                else:
                    candidate.status = "queued_for_rescue"
                    queued_for_rescue.append(int(candidate.wire_number))
                    repo.enqueue_rescue(
                        RescueQueueItem(
                            queue_id=f"{repo.session_id}:wire:{candidate.wire_number}",
                            wire_number=int(candidate.wire_number),
                            reason="low_confidence_or_ambiguous",
                        )
                    )
                repo.upsert_wire_candidate(candidate)

            return {
                "session_id": repo.session_id,
                "accepted_wires": [result.wire_number for result in accepted_results],
                "queued_wires": queued_for_rescue,
                "candidate_count": len(candidates),
            }
        finally:
            audio_stream.stop()
            self.pitch_worker.join()
            self._process_pitch_results(self.pitch_worker.drain_results())
            repo.close()
            self._active_repo = None
            self._active_session_id = None
            self._active_audio_stream = None

    def run_rescue(
        self,
        wire_number: int,
        *,
        session_id: str | None = None,
    ) -> dict[str, object]:
        repo = self.runtime.streaming_repository_factory(session_id)
        self._active_repo = repo
        self._active_session_id = repo.session_id
        predicted = self.runtime.wire_positions.get_true_position(
            apa_name=self.config.apa_name,
            layer=self.config.layer,
            side=self.config.side,
            flipped=self.config.flipped,
            wire_number=int(wire_number),
        )
        if predicted is None:
            repo.close()
            raise ValueError(f"No cached wire position available for wire {wire_number}")
        x_target, y_target = predicted
        zone = zone_lookup(x_target)
        length = length_lookup(self.config.layer, int(wire_number), zone)
        expected_frequency = wire_equation(length=length)["frequency"]
        direction = self.runtime.wire_positions.wire_direction(
            layer=self.config.layer,
            side=self.config.side,
            flipped=self.config.flipped,
        )
        trials = [
            (along, focus_offset)
            for along in (-self.config.rescue_position_step_mm, 0.0, self.config.rescue_position_step_mm)
            for focus_offset in (-self.config.focus_probe_step, 0.0, self.config.focus_probe_step)
        ]
        audio_stream = self.runtime.audio_stream_factory(
            self.config.sample_rate,
            self.config.hop_size,
        )
        audio_stream.start()
        best_candidate: WireCandidate | None = None
        try:
            for trial_index, (along_offset, focus_offset) in enumerate(trials):
                x_laser = float(x_target) + (float(direction[0]) * float(along_offset))
                y_laser = float(y_target) + (float(direction[1]) * float(along_offset))
                focus_reference = self.runtime.focus_plane.predict(x_laser, y_laser, clamp=False)
                if not np.isfinite(focus_reference):
                    focus_reference = float(self.config.default_focus)
                focus = float(focus_reference) + float(focus_offset)
                x_stage = stage_x_for_laser_target(
                    x_laser_target=x_laser,
                    focus=focus,
                    focus_reference=focus_reference,
                    side=self.config.side,
                )
                self._set_focus(focus)
                self._goto_stage_xy(x_stage, y_laser)
                start_time = self.runtime.clock()
                self.runtime.strum()
                time.sleep(float(self.config.rescue_capture_seconds))
                end_time = self.runtime.clock()
                segment = StreamingSegment(
                    segment_id=f"rescue:{wire_number}:{trial_index}",
                    mode="rescue",
                    pose0=build_measurement_pose(
                        x_true=x_stage,
                        y_true=y_laser,
                        focus=focus,
                        focus_reference=focus_reference,
                        side=self.config.side,
                    ),
                    pose1=build_measurement_pose(
                        x_true=x_stage,
                        y_true=y_laser,
                        focus=focus,
                        focus_reference=focus_reference,
                        side=self.config.side,
                    ),
                    speed_mm_s=0.0,
                    planned_start_time=start_time,
                    planned_end_time=end_time,
                    cruise_start_time=start_time,
                    cruise_end_time=end_time,
                    wire_hint=int(wire_number),
                    segment_status="completed",
                )
                repo.append_segment(segment)
                chunks = [
                    chunk
                    for chunk in audio_stream.drain_available()
                    if chunk.end_time >= start_time and chunk.start_time <= end_time
                ]
                self._analyze_chunks_for_segment(
                    repo=repo,
                    segment=segment,
                    chunks=chunks,
                    expected_frequency_hz=expected_frequency,
                    source_mode="rescue",
                    wire_hint=int(wire_number),
                )
                self.pitch_worker.join()
                self._process_pitch_results(self.pitch_worker.drain_results())

            candidates = self._aggregate_candidates(source_mode="rescue")
            if int(wire_number) in candidates:
                best_candidate = candidates[int(wire_number)]
                best_candidate.status = "accepted"
                repo.upsert_wire_candidate(best_candidate)
                self.runtime.focus_plane.add_anchor(
                    FocusAnchor(
                        anchor_id=f"{repo.session_id}:wire:{wire_number}",
                        x_true=best_candidate.best_pose.x_laser,
                        y_true=best_candidate.best_pose.y_true,
                        focus=best_candidate.best_pose.focus,
                        source="rescue",
                        pitch_hz=float(np.mean(best_candidate.pitch_estimates)),
                        confidence=float(np.mean(best_candidate.pitch_confidences)),
                    )
                )
                self.runtime.focus_plane.refit()
                result = self._persist_final_result(best_candidate, "stream_rescue")
                return {
                    "session_id": repo.session_id,
                    "wire_number": result.wire_number,
                    "frequency": result.frequency,
                    "confidence": result.confidence,
                }
            return {"session_id": repo.session_id, "wire_number": wire_number, "status": "no_result"}
        finally:
            audio_stream.stop()
            repo.close()
            self._active_repo = None
            self._active_session_id = None
