from __future__ import annotations

import importlib
from pathlib import Path
import sqlite3
import sys
import types

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

EXPECTED_COLUMNS = [
    "apa_name",
    "layer",
    "side",
    "wire_number",
    "frequency",
    "confidence",
    "x",
    "y",
    "time",
    "focus_position",
    "taped",
    "measurement_mode",
    "stream_session_id",
    "zone",
    "wire_length",
    "tension",
    "tension_pass",
]

def _load_services(monkeypatch):
    sys.modules.pop("dune_tension.services", None)
    results_stub = types.ModuleType("dune_tension.results")
    results_stub.EXPECTED_COLUMNS = EXPECTED_COLUMNS
    results_stub.TensionResult = object
    data_cache_stub = types.ModuleType("dune_tension.data_cache")
    data_cache_stub.append_dataframe_row = lambda *_args, **_kwargs: None
    data_cache_stub.append_results_row = lambda *_args, **_kwargs: None
    data_cache_stub.append_dataframe_rows = lambda *_args, **_kwargs: None
    data_cache_stub.append_results_rows = lambda *_args, **_kwargs: None
    data_cache_stub.connect_write_database = lambda _path: sqlite3.connect(":memory:")
    data_cache_stub.ensure_tables = lambda _conn: None
    monkeypatch.setitem(sys.modules, "dune_tension.results", results_stub)
    monkeypatch.setitem(sys.modules, "dune_tension.data_cache", data_cache_stub)
    return importlib.import_module("dune_tension.services")


def _make_result(expected_columns, **overrides):
    row = {column: None for column in expected_columns}
    row.update(
        {
            "apa_name": "APA",
            "layer": "X",
            "side": "A",
            "wire_number": 1,
            "frequency": 75.0,
            "confidence": 0.95,
            "x": 100.0,
            "y": 200.0,
            "taped": False,
            "time": "2026-03-14T10:00:00",
            "zone": 1,
            "wire_length": 1200.0,
            "tension": 6.0,
            "tension_pass": True,
        }
    )
    row.update(overrides)
    return types.SimpleNamespace(**row)


def test_result_repository_run_scope_batches_samples(monkeypatch) -> None:
    services = _load_services(monkeypatch)

    sample_batches: list[tuple[str, list[dict], dict]] = []
    result_batches: list[tuple[str, list[dict], dict]] = []

    monkeypatch.setattr(
        services.data_cache,
        "connect_write_database",
        lambda _path: sqlite3.connect(":memory:"),
    )
    monkeypatch.setattr(services.data_cache, "ensure_tables", lambda _conn: None)
    monkeypatch.setattr(
        services.data_cache,
        "append_results_rows",
        lambda path, rows, **kwargs: sample_batches.append((path, list(rows), kwargs)),
    )
    monkeypatch.setattr(
        services.data_cache,
        "append_dataframe_rows",
        lambda path, rows, **kwargs: result_batches.append((path, list(rows), kwargs)),
    )
    monkeypatch.setattr(
        services.data_cache,
        "append_results_row",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("run_scope should batch sample inserts")
        ),
    )
    monkeypatch.setattr(
        services.data_cache,
        "append_dataframe_row",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(
            AssertionError("run_scope should use shared write path for results")
        ),
    )

    repository = services.ResultRepository(
        "data/tension_data/tension_data.db",
        sample_batch_size=10,
    )

    with repository.run_scope():
        repository.append_sample(
            _make_result(EXPECTED_COLUMNS, wire_number=1)
        )
        repository.append_sample(
            _make_result(EXPECTED_COLUMNS, wire_number=2)
        )
        assert sample_batches == []

        repository.append_result(
            _make_result(EXPECTED_COLUMNS, wire_number=3)
        )
        assert len(result_batches) == 1
        assert [row["wire_number"] for row in result_batches[0][1]] == [3]

    assert len(sample_batches) == 1
    assert [row["wire_number"] for row in sample_batches[0][1]] == [1, 2]
