from dataclasses import KW_ONLY, dataclass, fields
from datetime import datetime
import math

from dune_tension.geometry import zone_lookup, length_lookup
from dune_tension.tension_calculation import tension_pass, wire_equation


@dataclass(frozen=True)
class DerivedTensionFields:
    zone: int
    wire_length: float
    tension: float
    tension_pass: bool


def derive_tension_fields(
    *,
    layer: str,
    wire_number: int,
    frequency: float,
    x: float,
    taped: bool = False,
) -> DerivedTensionFields:
    zone = zone_lookup(x)
    try:
        wire_length = length_lookup(layer, wire_number, zone, taped=taped)
        tension = wire_equation(wire_length, frequency)["tension"]
        tension_ok = tension_pass(tension, wire_length)
        if not math.isfinite(float(wire_length)) or not math.isfinite(float(tension)):
            raise ValueError("non-finite tension geometry")
        return DerivedTensionFields(
            zone=int(zone),
            wire_length=float(wire_length),
            tension=float(tension),
            tension_pass=bool(tension_ok),
        )
    except ValueError:
        return DerivedTensionFields(
            zone=int(zone),
            wire_length=0.0,
            tension=0.0,
            tension_pass=False,
        )


@dataclass
class TensionResult:
    apa_name: str
    layer: str
    side: str
    wire_number: int
    frequency: float
    confidence: float
    x: float
    y: float
    _: KW_ONLY
    time: datetime
    focus_position: int | None = None
    taped: bool = False
    measurement_mode: str = "legacy"
    stream_session_id: str | None = None
    zone: int = 0
    wire_length: float = 0.0
    tension: float = 0.0
    tension_pass: bool = False

    @classmethod
    def from_measurement(
        cls,
        *,
        apa_name: str,
        layer: str,
        side: str,
        wire_number: int,
        frequency: float,
        confidence: float,
        x: float,
        y: float,
        time: datetime,
        focus_position: int | None = None,
        taped: bool = False,
        measurement_mode: str = "legacy",
        stream_session_id: str | None = None,
    ) -> "TensionResult":
        derived = derive_tension_fields(
            layer=layer,
            wire_number=wire_number,
            frequency=frequency,
            x=x,
            taped=taped,
        )
        return cls(
            apa_name=apa_name,
            layer=layer,
            side=side,
            wire_number=wire_number,
            frequency=frequency,
            confidence=confidence,
            x=x,
            y=y,
            time=time,
            focus_position=focus_position,
            taped=taped,
            measurement_mode=measurement_mode,
            stream_session_id=stream_session_id,
            zone=derived.zone,
            wire_length=derived.wire_length,
            tension=derived.tension,
            tension_pass=derived.tension_pass,
        )


EXPECTED_COLUMNS = [f.name for f in fields(TensionResult)]
