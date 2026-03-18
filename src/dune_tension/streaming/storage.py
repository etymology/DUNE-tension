from __future__ import annotations

from dataclasses import is_dataclass
from datetime import datetime
import json
from pathlib import Path
import sqlite3
import threading
import uuid
import wave

import numpy as np

from dune_tension.streaming.models import (
    AudioChunkRef,
    FocusAnchor,
    PitchEvidenceBin,
    PitchResult,
    PulseEvent,
    RescueQueueItem,
    StreamingFrame,
    StreamingManifest,
    StreamingSegment,
    VoicedWindow,
    WireCandidate,
    model_to_dict,
)


def make_stream_session_id(now: datetime | None = None) -> str:
    timestamp = (now or datetime.utcnow()).strftime("%Y%m%dT%H%M%S")
    return f"{timestamp}-{uuid.uuid4().hex[:8]}"


class StreamingSessionRepository:
    """Append-heavy storage for one streaming measurement session."""

    def __init__(
        self,
        *,
        root_dir: str | Path = "data/streaming_runs",
        session_id: str | None = None,
    ) -> None:
        self.root_dir = Path(root_dir)
        self.root_dir.mkdir(parents=True, exist_ok=True)
        self.session_id = session_id or make_stream_session_id()
        self.session_dir = self.root_dir / self.session_id
        self.audio_dir = self.session_dir / "audio"
        self.session_dir.mkdir(parents=True, exist_ok=True)
        self.audio_dir.mkdir(parents=True, exist_ok=True)
        self.db_path = self.session_dir / "streaming.db"
        self.manifest_path = self.session_dir / "manifest.json"
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._ensure_schema()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _ensure_schema(self) -> None:
        with self._lock:
            self._conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS segments (
                    segment_id TEXT PRIMARY KEY,
                    mode TEXT,
                    status TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pulse_events (
                    pulse_id TEXT PRIMARY KEY,
                    segment_id TEXT,
                    timestamp REAL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS frames (
                    frame_id TEXT PRIMARY KEY,
                    segment_id TEXT,
                    timestamp REAL,
                    voiced_gate_pass INTEGER,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS voiced_windows (
                    window_id TEXT PRIMARY KEY,
                    segment_id TEXT,
                    start_time REAL,
                    end_time REAL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pitch_results (
                    pitch_result_id TEXT PRIMARY KEY,
                    window_id TEXT,
                    frequency_hz REAL,
                    confidence REAL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS pitch_bins (
                    bin_id TEXT PRIMARY KEY,
                    x_bin REAL,
                    y_bin REAL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS wire_candidates (
                    wire_number INTEGER PRIMARY KEY,
                    status TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS rescue_queue (
                    queue_id TEXT PRIMARY KEY,
                    wire_number INTEGER,
                    status TEXT,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS anchors (
                    anchor_id TEXT PRIMARY KEY,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS audio_chunks (
                    chunk_id TEXT PRIMARY KEY,
                    segment_id TEXT,
                    start_time REAL,
                    end_time REAL,
                    sample_rate INTEGER,
                    file_path TEXT,
                    payload_json TEXT NOT NULL
                );
                """
            )
            self._conn.commit()

    def _payload_json(self, value: object) -> str:
        if is_dataclass(value):
            return json.dumps(model_to_dict(value), sort_keys=True)
        return json.dumps(value, sort_keys=True)

    def _upsert(
        self,
        *,
        table: str,
        key_column: str,
        columns: dict[str, object],
    ) -> None:
        column_names = ", ".join(columns.keys())
        placeholders = ", ".join("?" for _ in columns)
        updates = ", ".join(
            f"{name}=excluded.{name}"
            for name in columns.keys()
            if name != key_column
        )
        sql = (
            f"INSERT INTO {table} ({column_names}) VALUES ({placeholders}) "
            f"ON CONFLICT({key_column}) DO UPDATE SET {updates}"
        )
        with self._lock:
            self._conn.execute(sql, tuple(columns.values()))
            self._conn.commit()

    def write_manifest(self, manifest: StreamingManifest) -> None:
        with self._lock:
            self.manifest_path.write_text(
                json.dumps(model_to_dict(manifest), indent=2, sort_keys=True) + "\n",
                encoding="utf-8",
            )

    def append_segment(self, segment: StreamingSegment) -> None:
        self._upsert(
            table="segments",
            key_column="segment_id",
            columns={
                "segment_id": segment.segment_id,
                "mode": segment.mode,
                "status": segment.segment_status,
                "payload_json": self._payload_json(segment),
            },
        )

    def append_pulse_event(self, event: PulseEvent) -> None:
        self._upsert(
            table="pulse_events",
            key_column="pulse_id",
            columns={
                "pulse_id": event.pulse_id,
                "segment_id": event.segment_id,
                "timestamp": float(event.timestamp),
                "payload_json": self._payload_json(event),
            },
        )

    def append_frame(self, frame: StreamingFrame) -> None:
        self._upsert(
            table="frames",
            key_column="frame_id",
            columns={
                "frame_id": frame.frame_id,
                "segment_id": frame.segment_id,
                "timestamp": float(frame.timestamp),
                "voiced_gate_pass": int(bool(frame.voiced_gate_pass)),
                "payload_json": self._payload_json(frame),
            },
        )

    def append_voiced_window(self, window_record: VoicedWindow) -> None:
        self._upsert(
            table="voiced_windows",
            key_column="window_id",
            columns={
                "window_id": window_record.window_id,
                "segment_id": window_record.segment_id,
                "start_time": float(window_record.start_time),
                "end_time": float(window_record.end_time),
                "payload_json": self._payload_json(window_record),
            },
        )

    def append_pitch_result(self, result: PitchResult) -> None:
        self._upsert(
            table="pitch_results",
            key_column="pitch_result_id",
            columns={
                "pitch_result_id": result.pitch_result_id,
                "window_id": result.window_id,
                "frequency_hz": float(result.frequency_hz),
                "confidence": float(result.confidence),
                "payload_json": self._payload_json(result),
            },
        )

    def upsert_pitch_bin(self, pitch_bin: PitchEvidenceBin) -> None:
        self._upsert(
            table="pitch_bins",
            key_column="bin_id",
            columns={
                "bin_id": pitch_bin.bin_id,
                "x_bin": float(pitch_bin.x_bin),
                "y_bin": float(pitch_bin.y_bin),
                "payload_json": self._payload_json(pitch_bin),
            },
        )

    def upsert_wire_candidate(self, candidate: WireCandidate) -> None:
        self._upsert(
            table="wire_candidates",
            key_column="wire_number",
            columns={
                "wire_number": int(candidate.wire_number),
                "status": candidate.status,
                "payload_json": self._payload_json(candidate),
            },
        )

    def enqueue_rescue(self, item: RescueQueueItem) -> None:
        self._upsert(
            table="rescue_queue",
            key_column="queue_id",
            columns={
                "queue_id": item.queue_id,
                "wire_number": int(item.wire_number),
                "status": item.status,
                "payload_json": self._payload_json(item),
            },
        )

    def append_anchor(self, anchor: FocusAnchor) -> None:
        self._upsert(
            table="anchors",
            key_column="anchor_id",
            columns={
                "anchor_id": anchor.anchor_id,
                "payload_json": self._payload_json(anchor),
            },
        )

    def append_audio_chunk(
        self,
        *,
        audio: np.ndarray,
        sample_rate: int,
        start_time: float,
        end_time: float,
        segment_id: str | None = None,
        chunk_id: str | None = None,
    ) -> AudioChunkRef:
        active_chunk_id = chunk_id or uuid.uuid4().hex
        file_name = f"{active_chunk_id}.wav"
        file_path = self.audio_dir / file_name
        self._write_pcm16_wav(file_path, audio, sample_rate)
        ref = AudioChunkRef(
            chunk_id=active_chunk_id,
            segment_id=segment_id,
            file_path=str(file_path),
            start_time=float(start_time),
            end_time=float(end_time),
            sample_rate=int(sample_rate),
        )
        self._upsert(
            table="audio_chunks",
            key_column="chunk_id",
            columns={
                "chunk_id": ref.chunk_id,
                "segment_id": ref.segment_id,
                "start_time": ref.start_time,
                "end_time": ref.end_time,
                "sample_rate": ref.sample_rate,
                "file_path": ref.file_path,
                "payload_json": self._payload_json(ref),
            },
        )
        return ref

    def _write_pcm16_wav(self, file_path: Path, audio: np.ndarray, sample_rate: int) -> None:
        data = np.asarray(audio, dtype=np.float32).reshape(-1)
        clipped = np.clip(data, -1.0, 1.0)
        pcm16 = np.round(clipped * 32767.0).astype(np.int16)
        with wave.open(str(file_path), "wb") as wav_file:
            wav_file.setnchannels(1)
            wav_file.setsampwidth(2)
            wav_file.setframerate(int(sample_rate))
            wav_file.writeframes(pcm16.tobytes())
