from __future__ import annotations

import math

from dune_tension.streaming.models import MeasurementPose, StreamingSegment

try:
    from dune_tension.tensiometer import FOCUS_X_MM_PER_QUARTER_US
except Exception:  # pragma: no cover - fallback for isolated imports
    FOCUS_X_MM_PER_QUARTER_US = (20.0 / 4000.0) / math.sqrt(3.0)


def focus_side_sign(side: str) -> float:
    """Return the sign for focus-induced X motion on the requested side."""

    return -1.0 if str(side).upper() == "A" else 1.0


def focus_to_x_delta_mm(delta_focus_units: float, side: str) -> float:
    """Translate focus error into the equivalent X correction in millimeters."""

    return (
        focus_side_sign(side)
        * float(delta_focus_units)
        * float(FOCUS_X_MM_PER_QUARTER_US)
    )


def build_measurement_pose(
    *,
    x_true: float,
    y_true: float,
    focus: float,
    focus_reference: float,
    side: str,
) -> MeasurementPose:
    """Return a measurement pose using the shared focus/X coupling transform."""

    x_focus_correction = focus_to_x_delta_mm(float(focus) - float(focus_reference), side)
    return MeasurementPose(
        x_true=float(x_true),
        y_true=float(y_true),
        focus=float(focus),
        focus_reference=float(focus_reference),
        x_focus_correction=float(x_focus_correction),
        x_laser=float(x_true + x_focus_correction),
        side=str(side).upper(),
    )


def interpolate_segment_pose(
    segment: StreamingSegment,
    timestamp: float,
) -> tuple[MeasurementPose, bool]:
    """Interpolate a pose for ``timestamp`` and flag whether it lies in cruise."""

    duration = max(float(segment.planned_end_time) - float(segment.planned_start_time), 1e-9)
    raw_ratio = (float(timestamp) - float(segment.planned_start_time)) / duration
    ratio = min(1.0, max(0.0, raw_ratio))

    x_true = segment.pose0.x_true + (segment.pose1.x_true - segment.pose0.x_true) * ratio
    y_true = segment.pose0.y_true + (segment.pose1.y_true - segment.pose0.y_true) * ratio
    focus = segment.pose0.focus + (segment.pose1.focus - segment.pose0.focus) * ratio
    focus_reference = segment.pose0.focus_reference + (
        segment.pose1.focus_reference - segment.pose0.focus_reference
    ) * ratio

    return (
        build_measurement_pose(
            x_true=x_true,
            y_true=y_true,
            focus=focus,
            focus_reference=focus_reference,
            side=segment.pose0.side,
        ),
        float(segment.cruise_start_time) <= float(timestamp) <= float(segment.cruise_end_time),
    )
