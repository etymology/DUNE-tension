from dataclasses import dataclass, field, fields
from datetime import datetime
import math

try:  # pragma: no cover - fallback for legacy test stubs
    from dune_tension.geometry import zone_lookup, length_lookup
    from dune_tension.tension_calculation import tension_pass, wire_equation
except ImportError:  # pragma: no cover
    from geometry import zone_lookup, length_lookup
    from tension_calculation import tension_pass, wire_equation


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
    focus_position: int | None = None
    taped: bool = False
    time: datetime | None = None
    measurement_mode: str = "legacy"
    stream_session_id: str | None = None

    zone: int = field(init=False)
    wire_length: float = field(init=False)
    tension: float = field(init=False)
    tension_pass: bool = field(init=False)

    def __post_init__(self) -> None:
        self.zone = zone_lookup(self.x)
        try:
            self.wire_length = length_lookup(
                self.layer, self.wire_number, self.zone, taped=self.taped
            )
            self.tension = wire_equation(self.wire_length, self.frequency)["tension"]
            self.tension_pass = tension_pass(self.tension, self.wire_length)
            if not math.isfinite(float(self.wire_length)) or not math.isfinite(
                float(self.tension)
            ):
                raise ValueError("non-finite tension geometry")
        except ValueError:
            self.wire_length = 0
            self.tension = 0
            self.tension_pass = False
        if self.time is None:
            self.time = datetime.now()


EXPECTED_COLUMNS = [f.name for f in fields(TensionResult)]
