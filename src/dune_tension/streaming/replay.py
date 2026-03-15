from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
import wave

import numpy as np

from dune_tension.streaming.analysis import FastFrameAnalyzer, StreamingAnalysisConfig
from dune_tension.streaming.pose import build_measurement_pose
from dune_tension.streaming.models import StreamingSegment
from dune_tension.streaming.runtime import TimedAudioChunk
from spectrum_analysis.pesto_analysis import analyze_audio_with_pesto


@dataclass(frozen=True)
class ReplaySummary:
    path: str
    sample_rate: int
    voiced_frame_count: int
    window_count: int
    max_comb_score: float
    best_frequency_hz: float | None
    best_confidence: float | None


def read_wav_mono(file_path: str | Path) -> tuple[int, np.ndarray]:
    path = Path(file_path)
    with wave.open(str(path), "rb") as wav_file:
        sample_rate = wav_file.getframerate()
        channels = wav_file.getnchannels()
        sample_width = wav_file.getsampwidth()
        frame_count = wav_file.getnframes()
        raw = wav_file.readframes(frame_count)
    if sample_width != 2:
        raise ValueError(f"Unsupported WAV sample width {sample_width} for {path}")
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    if channels > 1:
        audio = audio.reshape(-1, channels).mean(axis=1)
    return sample_rate, audio


def analyze_wav_file(
    file_path: str | Path,
    *,
    expected_frequency_hz: float | None = None,
) -> ReplaySummary:
    sample_rate, audio = read_wav_mono(file_path)
    analysis = FastFrameAnalyzer(StreamingAnalysisConfig(sample_rate=sample_rate))
    chunk = TimedAudioChunk(
        audio=audio,
        start_time=0.0,
        end_time=float(audio.size / sample_rate),
        sample_rate=sample_rate,
    )
    pose = build_measurement_pose(
        x_true=0.0,
        y_true=0.0,
        focus=4000.0,
        focus_reference=4000.0,
        side="A",
    )
    segment = StreamingSegment(
        segment_id=Path(file_path).stem,
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
    frames = analysis.analyze_chunk(
        chunk,
        segment=segment,
        expected_frequency_hz=expected_frequency_hz,
    )
    windows = analysis.build_voiced_windows(
        chunk,
        frames=frames,
        segment_id=segment.segment_id,
        audio_chunk_ref=None,
        expected_frequency_hz=expected_frequency_hz,
        source_mode="sweep",
    )
    best_frequency: float | None = None
    best_confidence: float | None = None
    for window in windows:
        result = analyze_audio_with_pesto(
            window.audio,
            sample_rate,
            expected_frequency=expected_frequency_hz,
            include_activations=False,
        )
        confidence = float(getattr(result, "confidence", float("nan")))
        if not np.isfinite(confidence):
            continue
        if best_confidence is None or confidence > best_confidence:
            best_confidence = confidence
            best_frequency = float(getattr(result, "frequency", float("nan")))

    return ReplaySummary(
        path=str(file_path),
        sample_rate=sample_rate,
        voiced_frame_count=sum(int(frame.voiced_gate_pass) for frame in frames),
        window_count=len(windows),
        max_comb_score=max((float(frame.comb_score) for frame in frames), default=0.0),
        best_frequency_hz=best_frequency,
        best_confidence=best_confidence,
    )


def analyze_wav_paths(
    paths: list[str | Path],
    *,
    expected_frequency_hz: float | None = None,
) -> list[ReplaySummary]:
    return [
        analyze_wav_file(path, expected_frequency_hz=expected_frequency_hz)
        for path in paths
    ]


def iter_wav_paths(path: str | Path) -> list[Path]:
    input_path = Path(path)
    if input_path.is_dir():
        return sorted(input_path.rglob("*.wav"))
    return [input_path]


def write_summary_csv(summaries: list[ReplaySummary], output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "path",
                "sample_rate",
                "voiced_frame_count",
                "window_count",
                "max_comb_score",
                "best_frequency_hz",
                "best_confidence",
            ],
        )
        writer.writeheader()
        for summary in summaries:
            writer.writerow(summary.__dict__)
