from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - dependency optional in tests
    pytest.skip("pandas is required for summaries tests", allow_module_level=True)

from dune_tension import summaries
from dune_tension.results import EXPECTED_COLUMNS


def test_order_missing_wires_basic() -> None:
    missing = [8, 1, 5, 7]
    measured = [10]
    ordered = summaries._order_missing_wires(missing, measured)
    assert ordered == [8, 7, 5, 1]


def test_order_missing_no_measured() -> None:
    missing = [3, 1, 2]
    ordered = summaries._order_missing_wires(missing, [])
    assert ordered == [1, 2, 3]


def test_get_missing_wires_from_database(tmp_path, monkeypatch) -> None:
    db_path = tmp_path / "data" / "tension_data" / "tension_data.db"
    db_path.parent.mkdir(parents=True)

    base_row = {
        "apa_name": "APA",
        "layer": "X",
        "side": "A",
        "wire_number": 1,
        "frequency": 100.0,
        "confidence": 0.9,
        "x": 0.0,
        "y": 0.0,
        "time": "2024-01-01T00:00:00",
        "zone": 0,
        "wire_length": 0.0,
        "tension": 1.0,
        "tension_pass": True,
    }

    rows = [
        base_row,
        {**base_row, "side": "B", "wire_number": 2, "time": "2024-01-01T00:01:00"},
    ]

    df = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    with sqlite3.connect(db_path) as conn:
        df.to_sql("tension_data", conn, if_exists="replace", index=False)

    monkeypatch.setattr(summaries, "get_expected_range", lambda _layer: range(1, 3))

    config = SimpleNamespace(
        apa_name="APA",
        layer="X",
        data_path=str(db_path),
        samples_per_wire=1,
        confidence_threshold=0.0,
    )

    missing = summaries.get_missing_wires(config)

    assert missing["A"] == [2]
    assert missing["B"] == [1]
