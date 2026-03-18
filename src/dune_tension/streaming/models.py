from __future__ import annotations

from dataclasses import asdict, dataclass, field, is_dataclass
from datetime import UTC, datetime
from typing import Any, Literal

MeasurementMode = Literal["legacy", "stream_sweep", "stream_rescue"]
StreamingSegmentMode = Literal["sweep", "rescue"]
StreamingSegmentStatus = Literal["planned", "running", "completed", "failed", "skipped"]
WireCandidateStatus = Literal[
    "provisional",
    "queued_for_rescue",
    "accepted",
    "rejected",
]
RescueQueueStatus = Literal["queued", "running", "completed", "failed", "cancelled"]


def _utc_now() -> datetime:
    return datetime.now(UTC)


@dataclass(frozen=True)
class MeasurementPose:
    x_true: float
    y_true: float
    focus: float
    focus_reference: float
    x_focus_correction: float
    x_laser: float
    side: str

    @property
    def delta_focus(self) -> float:
        return float(self.focus - self.focus_reference)


@dataclass(frozen=True)
class StreamingSegment:
    segment_id: str
    mode: StreamingSegmentMode
    pose0: MeasurementPose
    pose1: MeasurementPose
    speed_mm_s: float
    planned_start_time: float
    planned_end_time: float
    cruise_start_time: float
    cruise_end_time: float
    wire_hint: int | None = None
    segment_status: StreamingSegmentStatus = "planned"


@dataclass(frozen=True)
class AudioChunkRef:
    chunk_id: str
    segment_id: str | None
    file_path: str
    start_time: float
    end_time: float
    sample_rate: int


@dataclass(frozen=True)
class StreamingFrame:
    frame_id: str
    segment_id: str | None
    timestamp: float
    pose: MeasurementPose
    rms: float
    comb_score: float
    spectral_flatness: float
    harmonic_valid: bool
    expected_band_score: float | None
    voiced_gate_pass: bool
    audio_chunk_ref: str | None = None


@dataclass(frozen=True)
class VoicedWindow:
    window_id: str
    segment_id: str | None
    start_time: float
    end_time: float
    pose_center: MeasurementPose
    wire_hint: int | None = None
    audio_chunk_refs: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FocusResponsePoint:
    delta_focus: float
    support_count: int
    combined_confidence: float


@dataclass(frozen=True)
class PitchObservation:
    observation_id: str
    x_true: float
    y_true: float
    frequency_hz: float
    confidence: float
    comb_score: float
    focus_delta: float
    source_window_id: str | None = None
    source_sweep_id: str | None = None
    timestamp: datetime = field(default_factory=_utc_now)


@dataclass
class PitchHypothesis:
    pitch_center_hz: float
    support_count: int = 0
    weighted_pitch_hz: float = 0.0
    combined_confidence: float = 0.0
    max_pitch_confidence: float = 0.0
    max_comb_score: float = 0.0
    focus_response: list[FocusResponsePoint] = field(default_factory=list)


@dataclass
class PitchEvidenceBin:
    bin_id: str
    x_bin: float
    y_bin: float
    bin_size_mm: float
    hypotheses: list[PitchHypothesis] = field(default_factory=list)
    source_window_count: int = 0
    source_sweep_ids: list[str] = field(default_factory=list)
    source_window_ids: list[str] = field(default_factory=list)
    last_updated: datetime = field(default_factory=_utc_now)


@dataclass(frozen=True)
class PitchResult:
    pitch_result_id: str
    window_id: str
    frequency_hz: float
    confidence: float
    expected_frequency_hz: float | None
    frame_count: int
    source_mode: StreamingSegmentMode


@dataclass
class WireCandidate:
    wire_number: int
    source_mode: StreamingSegmentMode
    support_count: int
    best_pose: MeasurementPose
    best_comb_score: float
    pitch_estimates: list[float] = field(default_factory=list)
    pitch_confidences: list[float] = field(default_factory=list)
    angle_coherence_score: float = 0.0
    focus_consistency_score: float = 0.0
    status: WireCandidateStatus = "provisional"
    stream_session_id: str | None = None


@dataclass(frozen=True)
class FocusAnchor:
    anchor_id: str
    x_true: float
    y_true: float
    focus: float
    source: str = "manual"
    pitch_hz: float | None = None
    confidence: float | None = None


@dataclass(frozen=True)
class PulseEvent:
    pulse_id: str
    segment_id: str | None
    timestamp: float
    duration_s: float


@dataclass(frozen=True)
class RescueQueueItem:
    queue_id: str
    wire_number: int
    status: RescueQueueStatus = "queued"
    seed_segment_id: str | None = None
    seed_window_id: str | None = None
    reason: str = ""


@dataclass(frozen=True)
class StreamingManifest:
    session_id: str
    measurement_mode: MeasurementMode
    apa_name: str
    layer: str
    side: str
    flipped: bool
    runtime_config: dict[str, Any] = field(default_factory=dict)
    focus_plane_coefficients: dict[str, float] = field(default_factory=dict)
    anchors: list[dict[str, Any]] = field(default_factory=list)
    sweep_corridors: list[dict[str, Any]] = field(default_factory=list)
    code_version: str | None = None


@dataclass(frozen=True)
class PredictedWire:
    wire_number: int
    x_true: float
    y_true: float


def model_to_dict(value: Any) -> Any:
    if isinstance(value, datetime):
        return value.isoformat()
    if is_dataclass(value):
        return {key: model_to_dict(item) for key, item in asdict(value).items()}
    if isinstance(value, dict):
        return {str(key): model_to_dict(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [model_to_dict(item) for item in value]
    return value
