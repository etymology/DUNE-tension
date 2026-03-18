import sys
from pathlib import Path
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

# Provide a minimal 'requests' module for import
sys.modules.setdefault("requests", types.ModuleType("requests"))

import dune_tension.plc_io as plc
from dune_tension.config import GEOMETRY_CONFIG


def _setup(
    monkeypatch,
    start_x: float = 2000.0,
    start_y: float = float(GEOMETRY_CONFIG.y_min),
):
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
    plc.goto_xy(3500.0, float(GEOMETRY_CONFIG.y_max))
    assert moves == [
        (2000.0, float(GEOMETRY_CONFIG.y_min)),
        (3500.0, float(GEOMETRY_CONFIG.y_min)),
        (3500.0, float(GEOMETRY_CONFIG.y_max)),
    ]


def test_no_crossing_single_move(monkeypatch):
    moves = _setup(monkeypatch)
    plc.goto_xy(2100.0, float(GEOMETRY_CONFIG.y_max))
    assert moves == [(2100.0, float(GEOMETRY_CONFIG.y_max))]


def test_get_cached_xy_seeds_tracking_from_live_read(monkeypatch):
    monkeypatch.setattr(plc, "get_xy", lambda: (1234.5, 678.9))
    plc._TRUE_XY = [None, None]

    assert plc.get_cached_xy() == (1234.5, 678.9)
    assert plc._TRUE_XY == [1234.5, 678.9]
