import sys
from pathlib import Path
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Provide a minimal 'serial' module for import
sys.modules.setdefault("serial", types.ModuleType("serial"))

from dune_tension.maestro import DummyController


def test_dummy_servo():
    servo = DummyController()
    servo.setRange(0, 4000, 8000)
    servo.setTarget(0, 5000)
    assert servo.Targets[0] == 5000
    assert servo.getPosition(0) == 5000
    assert not servo.isMoving(0)
