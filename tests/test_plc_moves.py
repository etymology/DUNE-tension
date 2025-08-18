import sys
from pathlib import Path
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Provide a minimal 'requests' module for import
sys.modules.setdefault("requests", types.ModuleType("requests"))

import dune_tension.plc_io as plc


def _setup(monkeypatch, start_x: float = 2000.0, start_y: float = 500.0):
    moves = []
    cur = {"x": None, "y": None}

    def dummy_write_tag(name, value):
        if name == "X_POSITION":
            cur["x"] = value
        elif name == "Y_POSITION":
            cur["y"] = value
        elif name == "MOVE_TYPE" and value == plc.XY_MOVE_TYPE:
            moves.append((cur["x"], cur["y"]))
        return {}

    monkeypatch.setattr(plc, "write_tag", dummy_write_tag)
    monkeypatch.setattr(plc, "get_state", lambda: plc.IDLE_STATE)
    monkeypatch.setattr(plc, "get_movetype", lambda: plc.IDLE_MOVE_TYPE)
    monkeypatch.setattr(plc, "set_speed", lambda speed: True)
    monkeypatch.setattr(plc.time, "sleep", lambda s: None)
    plc._TRUE_XY = [start_x, start_y]
    plc._LAST_X_DIR = 0
    plc._X_DEADZONE_LEFT = 0.0
    return moves


def test_crossing_comb_splits_move(monkeypatch):
    moves = _setup(monkeypatch)
    plc.goto_xy(3500.0, 800.0)
    assert moves == [(2000.0, 0.0), (3500.0, 0.0), (3500.0, 800.0)]


def test_no_crossing_single_move(monkeypatch):
    moves = _setup(monkeypatch)
    plc.goto_xy(2100.0, 800.0)
    assert moves == [(2100.0, 800.0)]
