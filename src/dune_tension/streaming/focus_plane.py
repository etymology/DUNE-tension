from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from dune_tension.streaming.models import FocusAnchor


@dataclass
class FocusPlaneModel:
    """Planar focus model with optional servo clamping."""

    a: float = 0.0
    b: float = 0.0
    c: float = 0.0
    min_focus: float | None = None
    max_focus: float | None = None
    anchors: list[FocusAnchor] = field(default_factory=list)

    def coefficients(self) -> dict[str, float]:
        return {"a": float(self.a), "b": float(self.b), "c": float(self.c)}

    def predict(self, x_true: float, y_true: float, *, clamp: bool = True) -> float:
        focus = float(self.a * float(x_true) + self.b * float(y_true) + self.c)
        return self.clamp_focus(focus) if clamp else focus

    def clamp_focus(self, focus: float) -> float:
        if self.min_focus is not None:
            focus = max(float(self.min_focus), float(focus))
        if self.max_focus is not None:
            focus = min(float(self.max_focus), float(focus))
        return float(focus)

    def add_anchor(self, anchor: FocusAnchor) -> None:
        self.anchors.append(anchor)

    def extend_anchors(self, anchors: Iterable[FocusAnchor]) -> None:
        self.anchors.extend(anchors)

    def refit(self) -> bool:
        """Fit a plane from retained anchors, falling back to a constant focus."""

        if not self.anchors:
            return False

        if len(self.anchors) < 3:
            focus_values = [float(anchor.focus) for anchor in self.anchors]
            self.a = 0.0
            self.b = 0.0
            self.c = float(np.mean(focus_values))
            return True

        points = np.array(
            [
                [float(anchor.x_true), float(anchor.y_true), 1.0]
                for anchor in self.anchors
            ],
            dtype=np.float64,
        )
        values = np.array(
            [float(anchor.focus) for anchor in self.anchors],
            dtype=np.float64,
        )
        coeffs, *_ = np.linalg.lstsq(points, values, rcond=None)
        self.a = float(coeffs[0])
        self.b = float(coeffs[1])
        self.c = float(coeffs[2])
        return True

    @classmethod
    def fit_from_anchors(
        cls,
        anchors: Iterable[FocusAnchor],
        *,
        min_focus: float | None = None,
        max_focus: float | None = None,
    ) -> "FocusPlaneModel":
        model = cls(min_focus=min_focus, max_focus=max_focus)
        model.extend_anchors(list(anchors))
        model.refit()
        return model
