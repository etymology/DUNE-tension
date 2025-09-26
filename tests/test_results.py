import sys
import types
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Minimal numpy stub
numpy_stub = types.ModuleType("numpy")
numpy_stub.std = lambda arr: 0.0
sys.modules.setdefault("numpy", numpy_stub)

# geometry stub raising ValueError for length lookup
geo_stub = types.ModuleType("geometry")
geo_stub.zone_lookup = lambda x: 1


def _raise(*_a, **_k):
    raise ValueError("bad")


geo_stub.length_lookup = _raise
sys.modules["geometry"] = geo_stub

# tension_calculation stub
tc_stub = types.ModuleType("tension_calculation")
tc_stub.tension_lookup = lambda l, f: 0.0
tc_stub.tension_pass = lambda t, l: True
sys.modules["tension_calculation"] = tc_stub

from dune_tension.results import TensionResult


def test_value_error_defaults():
    res = TensionResult(
        apa_name="APA",
        layer="U",
        side="A",
        wire_number=1,
        frequency=10.0,
        confidence=0.5,
        x=0.0,
        y=0.0,
    )
    assert res.wire_length == 0
    assert res.tension == 0
    assert res.tension_pass is False
