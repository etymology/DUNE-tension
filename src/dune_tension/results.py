from dataclasses import dataclass, field, fields
from datetime import datetime

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
    time: datetime | None = None

    zone: int = field(init=False)
    wire_length: float = field(init=False)
    tension: float = field(init=False)
    tension_pass: bool = field(init=False)

    def __post_init__(self) -> None:
        self.zone = zone_lookup(self.x)
        try:
            self.wire_length = length_lookup(self.layer, self.wire_number, self.zone)
            self.tension = wire_equation(self.wire_length, self.frequency)["tension"]
            self.tension_pass = tension_pass(self.tension, self.wire_length)
        except ValueError:
            self.wire_length = 0
            self.tension = 0
            self.tension_pass = False
        if self.time is None:
            self.time = datetime.now()


EXPECTED_COLUMNS = [f.name for f in fields(TensionResult)]

