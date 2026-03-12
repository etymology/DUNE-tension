from __future__ import annotations

import sqlite3

import pytest

try:
    import pandas as pd
except ModuleNotFoundError:  # pragma: no cover - dependency optional in tests
    pytest.skip("pandas is required for data_cache tests", allow_module_level=True)

from dune_tension.data_cache import (
    append_dataframe_row,
    append_results_row,
    clear_wire_numbers,
    find_distribution_outliers,
    get_dataframe,
    get_results_dataframe,
)
from dune_tension.results import EXPECTED_COLUMNS


LEGACY_COLUMNS = [
    "apa_name",
    "layer",
    "side",
    "wire_number",
    "frequency",
    "confidence",
    "x",
    "y",
    "wires",
    "ttf",
    "time",
    "zone",
    "wire_length",
    "tension",
    "tension_pass",
    "t_sigma",
]


def _create_legacy_table(conn: sqlite3.Connection, table: str) -> None:
    columns_sql = ", ".join(f"{col} TEXT" for col in LEGACY_COLUMNS)
    conn.execute(f"CREATE TABLE {table} ({columns_sql})")


def test_append_row_migrates_legacy_db_schema(tmp_path) -> None:
    db_path = tmp_path / "legacy_tension_data.db"

    legacy_row = {
        "apa_name": "APA",
        "layer": "G",
        "side": "A",
        "wire_number": "4",
        "frequency": "74.1",
        "confidence": "0.95",
        "x": "1.0",
        "y": "2.0",
        "wires": "[]",
        "ttf": "0.0",
        "time": "2026-03-10T10:00:00",
        "zone": "1",
        "wire_length": "1200.0",
        "tension": "5.5",
        "tension_pass": "1",
        "t_sigma": "0.2",
    }

    with sqlite3.connect(db_path) as conn:
        _create_legacy_table(conn, "tension_data")
        _create_legacy_table(conn, "tension_samples")
        placeholders = ", ".join("?" for _ in LEGACY_COLUMNS)
        columns = ", ".join(LEGACY_COLUMNS)
        values = [legacy_row[col] for col in LEGACY_COLUMNS]
        conn.execute(
            f"INSERT INTO tension_data ({columns}) VALUES ({placeholders})", values
        )
        conn.execute(
            f"INSERT INTO tension_samples ({columns}) VALUES ({placeholders})", values
        )
        conn.commit()

    new_row = {
        "apa_name": "APA",
        "layer": "G",
        "side": "A",
        "wire_number": 5,
        "frequency": 75.9,
        "confidence": 1.0,
        "x": 6307.1064453125,
        "y": 352.9916076660156,
        "taped": True,
        "time": "2026-03-10T10:01:00",
        "zone": 1,
        "wire_length": 1285.0,
        "tension": 5.9,
        "tension_pass": True,
    }

    append_dataframe_row(str(db_path), new_row)
    append_results_row(str(db_path), new_row)

    with sqlite3.connect(db_path) as conn:
        tension_data_columns = [
            row[1] for row in conn.execute("PRAGMA table_info(tension_data)")
        ]
        tension_samples_columns = [
            row[1] for row in conn.execute("PRAGMA table_info(tension_samples)")
        ]

    assert "taped" in tension_data_columns
    assert "taped" in tension_samples_columns

    data_df = get_dataframe(str(db_path))
    results_df = get_results_dataframe(str(db_path))

    assert list(data_df.columns) == EXPECTED_COLUMNS
    assert list(results_df.columns) == EXPECTED_COLUMNS
    assert len(data_df) == 2
    assert len(results_df) == 2
    assert bool(data_df.iloc[-1]["taped"]) is True
    assert bool(results_df.iloc[-1]["taped"]) is True


def test_clear_wire_numbers_removes_selected_rows_from_both_tables(tmp_path) -> None:
    db_path = tmp_path / "tension_data.db"

    def make_row(wire_number: int, side: str = "A", time: str = "2026-03-10T10:00:00"):
        return {
            "apa_name": "APA",
            "layer": "G",
            "side": side,
            "wire_number": wire_number,
            "frequency": 75.0 + wire_number,
            "confidence": 0.95,
            "x": 100.0,
            "y": 200.0,
            "taped": False,
            "time": time,
            "zone": 1,
            "wire_length": 1200.0,
            "tension": 6.0 + wire_number,
            "tension_pass": True,
        }

    append_dataframe_row(str(db_path), make_row(1))
    append_dataframe_row(str(db_path), make_row(2))
    append_dataframe_row(str(db_path), make_row(3))
    append_dataframe_row(str(db_path), make_row(4, side="B"))

    append_results_row(str(db_path), make_row(1, time="2026-03-10T10:00:01"))
    append_results_row(str(db_path), make_row(2, time="2026-03-10T10:00:02"))
    append_results_row(str(db_path), make_row(2, time="2026-03-10T10:00:03"))
    append_results_row(str(db_path), make_row(3, time="2026-03-10T10:00:04"))
    append_results_row(str(db_path), make_row(4, side="B", time="2026-03-10T10:00:05"))

    clear_wire_numbers(str(db_path), "APA", "G", "A", [2, 99])

    data_df = get_dataframe(str(db_path))
    results_df = get_results_dataframe(str(db_path))

    remaining_data = sorted(
        (int(row.wire_number), row.side)
        for row in data_df.itertuples(index=False)
    )
    remaining_results = sorted(
        (int(row.wire_number), row.side)
        for row in results_df.itertuples(index=False)
    )

    assert remaining_data == [(1, "A"), (3, "A"), (4, "B")]
    assert remaining_results == [(1, "A"), (3, "A"), (4, "B")]


def test_find_distribution_outliers_uses_bulk_tension_distribution(tmp_path) -> None:
    db_path = tmp_path / "distribution_outliers.db"

    def make_row(wire_number: int, tension: float, confidence: float = 0.95) -> dict:
        return {
            "apa_name": "APA",
            "layer": "G",
            "side": "A",
            "wire_number": wire_number,
            "frequency": 75.0,
            "confidence": confidence,
            "x": 100.0,
            "y": 200.0,
            "taped": False,
            "time": "2026-03-10T10:00:00",
            "zone": 1,
            "wire_length": 1200.0,
            "tension": tension,
            "tension_pass": True,
        }

    for wire_number in range(1, 11):
        append_dataframe_row(str(db_path), make_row(wire_number, 5.0))
    append_dataframe_row(str(db_path), make_row(11, 9.0))
    append_dataframe_row(str(db_path), make_row(12, 9.5, confidence=0.1))

    outliers = find_distribution_outliers(
        str(db_path),
        "APA",
        "G",
        "A",
        times_sigma=2.0,
        confidence_threshold=0.9,
    )

    assert outliers == [11]
