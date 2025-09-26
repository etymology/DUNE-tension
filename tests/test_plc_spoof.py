import sys
from pathlib import Path
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Provide a minimal 'requests' module for import
sys.modules.setdefault("requests", types.ModuleType("requests"))

import dune_tension.plc_io as plc


def test_spoof_functions():
    # Basic behaviour
    # default spoofed position
    assert plc.spoof_get_xy() == (6300.0, 200.0)
    assert plc.spoof_goto_xy(10, 20)
    # Position should update
    assert plc.spoof_get_xy() == (10, 20)
    assert plc.spoof_wiggle(0.5)
    # Bounds check should warn but still succeed
    assert plc.spoof_goto_xy(-1, -1)
    # Position updated even for out-of-bounds
    assert plc.spoof_get_xy() == (-1, -1)
