from __future__ import annotations

from datetime import datetime
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import dune_tension.results as results_module
from dune_tension.results import TensionResult, derive_tension_fields


def test_derive_tension_fields_defaults_when_geometry_fails(monkeypatch) -> None:
    monkeypatch.setattr(results_module, "zone_lookup", lambda _x: 1)

    def _raise(*_args, **_kwargs):
        raise ValueError("bad")

    monkeypatch.setattr(results_module, "length_lookup", _raise)

    derived = derive_tension_fields(
        layer="U",
        wire_number=1,
        frequency=10.0,
        x=0.0,
    )

    assert derived.zone == 1
    assert derived.wire_length == 0.0
    assert derived.tension == 0.0
    assert derived.tension_pass is False


def test_tension_result_from_measurement_populates_derived_fields(monkeypatch) -> None:
    monkeypatch.setattr(results_module, "zone_lookup", lambda _x: 2)
    monkeypatch.setattr(
        results_module,
        "length_lookup",
        lambda _layer, _wire, _zone, taped=False: 1.5,
    )
    monkeypatch.setattr(
        results_module,
        "wire_equation",
        lambda length, frequency: {"tension": 6.0, "frequency": frequency},
    )
    monkeypatch.setattr(results_module, "tension_pass", lambda _tension, _length: True)

    timestamp = datetime(2026, 3, 15, 12, 0, 0)
    result = TensionResult.from_measurement(
        apa_name="APA",
        layer="U",
        side="A",
        wire_number=1,
        frequency=10.0,
        confidence=0.5,
        x=0.0,
        y=0.0,
        time=timestamp,
    )

    assert result.time is timestamp
    assert result.zone == 2
    assert result.wire_length == 1.5
    assert result.tension == 6.0
    assert result.tension_pass is True
