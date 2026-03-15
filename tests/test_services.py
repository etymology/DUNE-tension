from __future__ import annotations

import sqlite3
import types

from dune_tension.results import EXPECTED_COLUMNS
from dune_tension.services import ResultRepository


def _make_result(**overrides):
    row = {column: None for column in EXPECTED_COLUMNS}
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
    import dune_tension.services as services

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

    repository = ResultRepository("data/tension_data/tension_data.db", sample_batch_size=10)

    with repository.run_scope():
        repository.append_sample(_make_result(wire_number=1))
        repository.append_sample(_make_result(wire_number=2))
        assert sample_batches == []

        repository.append_result(_make_result(wire_number=3))
        assert len(result_batches) == 1
        assert [row["wire_number"] for row in result_batches[0][1]] == [3]

    assert len(sample_batches) == 1
    assert [row["wire_number"] for row in sample_batches[0][1]] == [1, 2]
