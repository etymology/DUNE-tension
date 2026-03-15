from __future__ import annotations

from dataclasses import replace
from typing import Iterable

from dune_tension.streaming.models import (
    FocusResponsePoint,
    PitchEvidenceBin,
    PitchHypothesis,
    PitchObservation,
)


def merge_pitch_confidence(current: float, new_value: float) -> float:
    """Fuse two independent confidence values monotonically."""

    cur = min(1.0, max(0.0, float(current)))
    new = min(1.0, max(0.0, float(new_value)))
    return float(1.0 - ((1.0 - cur) * (1.0 - new)))


class PitchEvidenceField:
    """Spatial bins that accumulate compatible pitch observations."""

    def __init__(
        self,
        *,
        bin_size_mm: float = 0.5,
        pitch_tolerance_hz: float = 5.0,
        pitch_tolerance_ratio: float = 0.01,
        focus_bucket_units: float = 25.0,
    ) -> None:
        self.bin_size_mm = float(bin_size_mm)
        self.pitch_tolerance_hz = float(pitch_tolerance_hz)
        self.pitch_tolerance_ratio = float(pitch_tolerance_ratio)
        self.focus_bucket_units = float(focus_bucket_units)
        self._bins: dict[tuple[int, int], PitchEvidenceBin] = {}

    def __len__(self) -> int:
        return len(self._bins)

    def bins(self) -> list[PitchEvidenceBin]:
        return list(self._bins.values())

    def _bin_key(self, x_true: float, y_true: float) -> tuple[int, int]:
        return (
            int(round(float(x_true) / self.bin_size_mm)),
            int(round(float(y_true) / self.bin_size_mm)),
        )

    def _bin_center(self, key: tuple[int, int]) -> tuple[float, float]:
        return (key[0] * self.bin_size_mm, key[1] * self.bin_size_mm)

    def _pitch_tolerance(self, lhs_hz: float, rhs_hz: float) -> float:
        return max(
            self.pitch_tolerance_hz,
            self.pitch_tolerance_ratio * max(abs(float(lhs_hz)), abs(float(rhs_hz))),
        )

    def _get_or_create_bin(self, observation: PitchObservation) -> PitchEvidenceBin:
        key = self._bin_key(observation.x_true, observation.y_true)
        current = self._bins.get(key)
        if current is not None:
            return current
        x_bin, y_bin = self._bin_center(key)
        created = PitchEvidenceBin(
            bin_id=f"{key[0]}:{key[1]}",
            x_bin=x_bin,
            y_bin=y_bin,
            bin_size_mm=self.bin_size_mm,
        )
        self._bins[key] = created
        return created

    def _match_hypothesis(
        self,
        hypotheses: Iterable[PitchHypothesis],
        frequency_hz: float,
    ) -> PitchHypothesis | None:
        for hypothesis in hypotheses:
            tolerance = self._pitch_tolerance(hypothesis.pitch_center_hz, frequency_hz)
            if abs(float(hypothesis.pitch_center_hz) - float(frequency_hz)) <= tolerance:
                return hypothesis
        return None

    def _bucket_focus_delta(self, focus_delta: float) -> float:
        bucket = round(float(focus_delta) / self.focus_bucket_units)
        return float(bucket * self.focus_bucket_units)

    def _update_focus_response(
        self,
        hypothesis: PitchHypothesis,
        *,
        focus_delta: float,
        confidence: float,
    ) -> None:
        target = self._bucket_focus_delta(focus_delta)
        for index, point in enumerate(hypothesis.focus_response):
            if float(point.delta_focus) != target:
                continue
            hypothesis.focus_response[index] = replace(
                point,
                support_count=int(point.support_count) + 1,
                combined_confidence=merge_pitch_confidence(
                    point.combined_confidence,
                    confidence,
                ),
            )
            return

        hypothesis.focus_response.append(
            FocusResponsePoint(
                delta_focus=target,
                support_count=1,
                combined_confidence=min(1.0, max(0.0, float(confidence))),
            )
        )
        hypothesis.focus_response.sort(key=lambda item: float(item.delta_focus))

    def observe(self, observation: PitchObservation) -> PitchEvidenceBin:
        evidence_bin = self._get_or_create_bin(observation)
        if observation.source_window_id and observation.source_window_id not in evidence_bin.source_window_ids:
            evidence_bin.source_window_ids.append(observation.source_window_id)
            evidence_bin.source_window_count = len(evidence_bin.source_window_ids)
        if observation.source_sweep_id and observation.source_sweep_id not in evidence_bin.source_sweep_ids:
            evidence_bin.source_sweep_ids.append(observation.source_sweep_id)
        evidence_bin.last_updated = observation.timestamp

        hypothesis = self._match_hypothesis(evidence_bin.hypotheses, observation.frequency_hz)
        if hypothesis is None:
            hypothesis = PitchHypothesis(
                pitch_center_hz=float(observation.frequency_hz),
                support_count=0,
                weighted_pitch_hz=float(observation.frequency_hz),
            )
            evidence_bin.hypotheses.append(hypothesis)

        hypothesis.support_count += 1
        total_weight = max(hypothesis.combined_confidence, 0.0) + max(
            float(observation.confidence),
            0.0,
        )
        if total_weight <= 0.0:
            hypothesis.weighted_pitch_hz = float(observation.frequency_hz)
        else:
            hypothesis.weighted_pitch_hz = (
                (hypothesis.weighted_pitch_hz * max(hypothesis.combined_confidence, 0.0))
                + (float(observation.frequency_hz) * max(float(observation.confidence), 0.0))
            ) / total_weight
        hypothesis.pitch_center_hz = float(hypothesis.weighted_pitch_hz)
        hypothesis.combined_confidence = merge_pitch_confidence(
            hypothesis.combined_confidence,
            observation.confidence,
        )
        hypothesis.max_pitch_confidence = max(
            float(hypothesis.max_pitch_confidence),
            float(observation.confidence),
        )
        hypothesis.max_comb_score = max(
            float(hypothesis.max_comb_score),
            float(observation.comb_score),
        )
        self._update_focus_response(
            hypothesis,
            focus_delta=observation.focus_delta,
            confidence=observation.confidence,
        )
        evidence_bin.hypotheses.sort(
            key=lambda item: (
                float(item.combined_confidence),
                float(item.max_pitch_confidence),
                float(item.max_comb_score),
            ),
            reverse=True,
        )
        return evidence_bin

    def dominant_hypothesis(
        self,
        x_true: float,
        y_true: float,
    ) -> PitchHypothesis | None:
        evidence_bin = self._bins.get(self._bin_key(x_true, y_true))
        if evidence_bin is None or not evidence_bin.hypotheses:
            return None
        return evidence_bin.hypotheses[0]

    def observe_many(self, observations: Iterable[PitchObservation]) -> None:
        for observation in observations:
            self.observe(observation)

    def snapshot(self) -> list[PitchEvidenceBin]:
        return sorted(
            self._bins.values(),
            key=lambda item: (float(item.x_bin), float(item.y_bin)),
        )
