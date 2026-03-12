import importlib
import sys
import types
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from dune_tension.config import GEOMETRY_CONFIG


def _load_module(monkeypatch):
    data_cache = types.ModuleType("dune_tension.data_cache")
    data_cache.get_dataframe = lambda _path: None
    monkeypatch.setitem(sys.modules, "dune_tension.data_cache", data_cache)

    plc_io = types.ModuleType("dune_tension.plc_io")
    plc_io.is_motion_target_in_bounds = (
        lambda x, y: (
            GEOMETRY_CONFIG.x_min <= float(x) <= GEOMETRY_CONFIG.x_max
            and GEOMETRY_CONFIG.y_min <= float(y) <= GEOMETRY_CONFIG.y_max
        )
    )
    monkeypatch.setitem(sys.modules, "dune_tension.plc_io", plc_io)

    sys.modules.pop("dune_tension.tensiometer_functions", None)
    return importlib.import_module("dune_tension.tensiometer_functions")


def test_plan_measurement_triplets_filters_illegal_targets_and_orders_greedily(
    monkeypatch, caplog
):
    tensiometer_functions = _load_module(monkeypatch)
    caplog.set_level("WARNING")
    config = types.SimpleNamespace()
    positions = {
        1: (4010.0, 1460.0),
        2: (4076.5, -1816.9),
        3: (4000.0, 1460.0),
        4: (4200.0, 1495.0),
    }

    planned = tensiometer_functions.plan_measurement_triplets(
        config=config,
        wire_list=[1, 2, 3, 4],
        get_xy_from_file_func=lambda _config, wire: positions[wire],
        get_current_xy_func=lambda: (4200.0, 1460.0),
        preserve_order=False,
    )

    assert planned == [
        (4, 4200.0, 1495.0),
        (1, 4010.0, 1460.0),
        (3, 4000.0, 1460.0),
    ]
    assert "Skipping wire 2 because motion target 4076.5,-1816.9 is out of bounds." in caplog.text


def test_measure_list_uses_shared_planner_output(monkeypatch):
    tensiometer_functions = _load_module(monkeypatch)
    collected = []
    planner_calls = []

    monkeypatch.setattr(
        tensiometer_functions,
        "plan_measurement_triplets",
        lambda **kwargs: planner_calls.append(kwargs) or [(8, 8.0, 80.0), (3, 3.0, 30.0)],
    )

    tensiometer_functions.measure_list(
        config=types.SimpleNamespace(),
        wire_list=[3, 8],
        get_xy_from_file_func=lambda *_args, **_kwargs: None,
        get_current_xy_func=lambda: (0.0, 0.0),
        collect_func=lambda wire, x, y: collected.append((wire, x, y)),
        preserve_order=False,
        profile=False,
    )

    assert len(planner_calls) == 1
    assert planner_calls[0]["wire_list"] == [3, 8]
    assert collected == [(8, 8.0, 80.0), (3, 3.0, 30.0)]
