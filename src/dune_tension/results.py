from dataclasses import dataclass, field, fields
from datetime import datetime
from typing import List

import numpy as np
from geometry import zone_lookup, length_lookup
from tension_calculation import tension_lookup, tension_pass


@dataclass
class TensionResult:
    apa_name: str
    layer: str
    side: str
    wire_number: int
    frequency: float = 0.0
    confidence: float = 0.0
    x: float = 0.0
    y: float = 0.0
    wires: List[float] | None = None
    ttf: float = 0.0
    time: datetime | None = None

    zone: int = field(init=False)
    wire_length: float = field(init=False)
    tension: float = field(init=False)
    tension_pass: bool = field(init=False)
    t_sigma: float = field(init=False)

    def __post_init__(self) -> None:
        self.wires = self.wires or []
        self.zone = zone_lookup(self.x)
        self.wire_length = length_lookup(self.layer, self.wire_number, self.zone)
        self.tension = tension_lookup(self.wire_length, self.frequency)
        self.tension_pass = tension_pass(self.tension, self.wire_length)
        self.t_sigma = float(np.std(self.wires)) if self.wires else 0.0
        if self.time is None:
            self.time = datetime.now()


EXPECTED_COLUMNS = [f.name for f in fields(TensionResult)]


@dataclass
class RawSample:
    """Individual sample recorded before clustering."""

    apa_name: str
    layer: str
    side: str
    wire_number: int
    frequency: float
    confidence: float
    x: float
    y: float
    time: str


RAW_SAMPLE_COLUMNS = [f.name for f in fields(RawSample)]
