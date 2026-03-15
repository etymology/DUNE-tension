from __future__ import annotations

import math
import sqlite3
import sys
from pathlib import Path

import numpy as np
import pytest

# Ensure src is on path.
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dune_tension.results import TensionResult
from dune_tension.streaming import (
    FocusAnchor,
    FocusPlaneModel,
    PitchEvidenceField,
    PitchObservation,
    StreamingFrame,
    StreamingManifest,
    StreamingSegment,
    StreamingSessionRepository,
    build_measurement_pose,
    focus_to_x_delta_mm,
    interpolate_segment_pose,
)
from spectrum_analysis.comb_trigger import harmonic_comb_response


def test_tension_result_defaults_to_legacy_measurement_mode() -> None:
    result = TensionResult(
        apa_name="APA",
        layer="X",
        side="A",
        wire_number=1,
        frequency=72.0,
        confidence=0.8,
        x=1.0,
        y=2.0,
    )

    assert result.measurement_mode == "legacy"
    assert result.stream_session_id is None


def test_interpolate_segment_pose_applies_focus_correction_and_cruise_mask() -> None:
    pose0 = build_measurement_pose(
        x_true=10.0,
        y_true=20.0,
        focus=4000,
        focus_reference=4000,
        side="A",
    )
    pose1 = build_measurement_pose(
        x_true=14.0,
        y_true=24.0,
        focus=4050,
        focus_reference=4000,
        side="A",
    )
    segment = StreamingSegment(
        segment_id="seg-1",
        mode="sweep",
        pose0=pose0,
        pose1=pose1,
        speed_mm_s=10.0,
        planned_start_time=0.0,
        planned_end_time=4.0,
        cruise_start_time=1.0,
        cruise_end_time=3.0,
    )

    pose_mid, in_cruise = interpolate_segment_pose(segment, 2.0)

    assert in_cruise is True
    assert pose_mid.x_true == 12.0
    assert pose_mid.y_true == 22.0
    assert pose_mid.focus == 4025.0
    assert pose_mid.x_focus_correction == focus_to_x_delta_mm(25.0, "A")
    assert pose_mid.x_laser == pose_mid.x_true + pose_mid.x_focus_correction

    _, before_cruise = interpolate_segment_pose(segment, 0.25)
    assert before_cruise is False


def test_pitch_evidence_field_merges_compatible_observations_monotonically() -> None:
    field = PitchEvidenceField(bin_size_mm=0.5)
    observation0 = PitchObservation(
        observation_id="obs-0",
        x_true=100.0,
        y_true=200.0,
        frequency_hz=110.0,
        confidence=0.4,
        comb_score=0.2,
        focus_delta=0.0,
        source_window_id="win-0",
        source_sweep_id="sweep-0",
    )
    observation1 = PitchObservation(
        observation_id="obs-1",
        x_true=100.1,
        y_true=199.9,
        frequency_hz=111.0,
        confidence=0.6,
        comb_score=0.8,
        focus_delta=25.0,
        source_window_id="win-1",
        source_sweep_id="sweep-1",
    )

    pitch_bin = field.observe(observation0)
    pitch_bin = field.observe(observation1)

    assert len(field) == 1
    assert pitch_bin.source_window_count == 2
    assert set(pitch_bin.source_sweep_ids) == {"sweep-0", "sweep-1"}
    assert len(pitch_bin.hypotheses) == 1

    hypothesis = pitch_bin.hypotheses[0]
    assert hypothesis.support_count == 2
    assert hypothesis.combined_confidence > 0.6
    assert math.isclose(hypothesis.max_comb_score, 0.8)
    assert len(hypothesis.focus_response) == 2


def test_focus_plane_model_fits_planar_surface() -> None:
    anchors = [
        FocusAnchor(anchor_id="a0", x_true=0.0, y_true=0.0, focus=4000.0),
        FocusAnchor(anchor_id="a1", x_true=1.0, y_true=0.0, focus=4010.0),
        FocusAnchor(anchor_id="a2", x_true=0.0, y_true=2.0, focus=4040.0),
        FocusAnchor(anchor_id="a3", x_true=1.0, y_true=2.0, focus=4050.0),
    ]

    model = FocusPlaneModel.fit_from_anchors(anchors)

    assert model.predict(0.5, 1.0, clamp=False) == pytest.approx(4025.0)


def test_streaming_session_repository_writes_manifest_and_session_tables(tmp_path) -> None:
    repo = StreamingSessionRepository(root_dir=tmp_path, session_id="session-1")
    manifest = StreamingManifest(
        session_id="session-1",
        measurement_mode="stream_sweep",
        apa_name="APA",
        layer="X",
        side="A",
        flipped=False,
        runtime_config={"speed_mm_s": 20.0},
    )
    repo.write_manifest(manifest)
    frame = StreamingFrame(
        frame_id="frame-1",
        segment_id="segment-1",
        timestamp=1.23,
        pose=build_measurement_pose(
            x_true=1.0,
            y_true=2.0,
            focus=4000,
            focus_reference=4000,
            side="A",
        ),
        rms=0.5,
        comb_score=0.7,
        spectral_flatness=0.2,
        harmonic_valid=True,
        expected_band_score=None,
        voiced_gate_pass=True,
    )
    repo.append_frame(frame)
    ref = repo.append_audio_chunk(
        audio=np.array([0.0, 0.25, -0.25], dtype=np.float32),
        sample_rate=16000,
        start_time=1.0,
        end_time=1.001,
        segment_id="segment-1",
    )
    repo.close()

    assert (tmp_path / "session-1" / "manifest.json").exists()
    assert (tmp_path / "session-1" / "audio" / f"{ref.chunk_id}.wav").exists()

    with sqlite3.connect(tmp_path / "session-1" / "streaming.db") as conn:
        frame_count = conn.execute("SELECT COUNT(*) FROM frames").fetchone()[0]
        audio_count = conn.execute("SELECT COUNT(*) FROM audio_chunks").fetchone()[0]

    assert frame_count == 1
    assert audio_count == 1


def test_harmonic_comb_response_detects_harmonic_signal() -> None:
    sample_rate = 16000
    frame_size = 2048
    t = np.arange(frame_size, dtype=np.float64) / sample_rate
    frame = (
        0.8 * np.sin(2.0 * np.pi * 220.0 * t)
        + 0.3 * np.sin(2.0 * np.pi * 440.0 * t)
        + 0.2 * np.sin(2.0 * np.pi * 660.0 * t)
    ).astype(np.float32)
    window = np.hanning(frame_size).astype(np.float32)
    freq_bins = np.fft.rfftfreq(frame_size, d=1.0 / sample_rate)
    candidates = np.geomspace(180.0, 260.0, num=24)
    weights = 1.0 / np.arange(1, 8, dtype=np.float64)

    comb_score, sfm, valid = harmonic_comb_response(
        frame,
        sample_rate,
        window,
        freq_bins,
        candidates,
        weights,
        min_harmonics=3,
    )

    assert valid is True
    assert comb_score > 0.1
    assert sfm < 0.6
