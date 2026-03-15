import importlib
import sys
import types
from pathlib import Path

import pytest

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


def test_wire_position_provider_caches_latest_positions(monkeypatch):
    import dune_tension.tensiometer_functions as tensiometer_functions

    class FakeArray(list):
        def __sub__(self, other):
            return FakeArray([value - other for value in self])

    class FakeMask(list):
        def __and__(self, other):
            return FakeMask([left and right for left, right in zip(self, other)])

    class FakeStringAccessor:
        def __init__(self, values):
            self._values = values

        def upper(self):
            return FakeSeries([str(value).upper() for value in self._values])

    class FakeSeries:
        def __init__(self, values):
            self._values = list(values)

        def __eq__(self, other):
            return FakeMask([value == other for value in self._values])

        def astype(self, dtype):
            converter = str if dtype is str else dtype
            return FakeSeries([converter(value) for value in self._values])

        @property
        def str(self):
            return FakeStringAccessor(self._values)

        @property
        def values(self):
            return FakeArray(self._values)

    class FakeDataFrame:
        def __init__(self, rows):
            self._rows = list(rows)

        def __getitem__(self, key):
            if isinstance(key, str):
                return FakeSeries([row[key] for row in self._rows])
            return FakeDataFrame(
                [row for row, keep in zip(self._rows, key) if keep]
            )

        @property
        def empty(self):
            return not self._rows

        def sort_values(self, column):
            return FakeDataFrame(sorted(self._rows, key=lambda row: row[column]))

        def drop_duplicates(self, subset, keep="last"):
            latest_rows = {}
            for row in self._rows:
                latest_rows[row[subset]] = row
            return FakeDataFrame(
                [row for row in self._rows if latest_rows[row[subset]] is row]
            )

        def reset_index(self, drop=True):
            return FakeDataFrame(self._rows)

    numpy_stub = sys.modules["numpy"]
    monkeypatch.setattr(
        numpy_stub,
        "abs",
        lambda arr: FakeArray([abs(value) for value in arr]),
        raising=False,
    )
    monkeypatch.setattr(
        numpy_stub,
        "argmin",
        lambda arr: min(range(len(arr)), key=lambda index: arr[index]),
        raising=False,
    )

    loader_calls = []
    df = FakeDataFrame(
        [
            {
                "apa_name": "APA",
                "layer": "X",
                "side": "A",
                "wire_number": 10,
                "x": 99.0,
                "y": 199.0,
                "time": "2026-03-10T10:00:00",
            },
            {
                "apa_name": "APA",
                "layer": "X",
                "side": "A",
                "wire_number": 10,
                "x": 100.0,
                "y": 200.0,
                "time": "2026-03-10T10:01:00",
            },
            {
                "apa_name": "APA",
                "layer": "X",
                "side": "A",
                "wire_number": 12,
                "x": 120.0,
                "y": 220.0,
                "time": "2026-03-10T10:02:00",
            },
        ]
    )

    provider = tensiometer_functions.WirePositionProvider(
        dataframe_loader=lambda _path: loader_calls.append(True) or df
    )
    config = tensiometer_functions.make_config(apa_name="APA", layer="X", side="A")

    xy_wire_10 = provider.get_xy(config, 10)
    xy_wire_12 = provider.get_xy(config, 12)

    assert loader_calls == [True]
    assert xy_wire_10 == (100.0, 200.0)
    assert xy_wire_12 == (120.0, 220.0)
