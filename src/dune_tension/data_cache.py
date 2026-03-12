from __future__ import annotations

import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

import pandas as pd

from dune_tension.results import EXPECTED_COLUMNS

TABLE_TENSION_DATA = "tension_data"
TABLE_TENSION_SAMPLES = "tension_samples"

# In-process DataFrame cache.
_dataframe_cache: dict[str, pd.DataFrame] = {}


def _table_columns_sql() -> str:
    return ", ".join(f"{col} TEXT" for col in EXPECTED_COLUMNS)


def _get_table_columns(conn: sqlite3.Connection, table: str) -> list[str]:
    rows = conn.execute(f"PRAGMA table_info({table})").fetchall()
    return [str(row[1]) for row in rows]


def _normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    normalized = df.copy()
    for col in EXPECTED_COLUMNS:
        if col not in normalized.columns:
            normalized[col] = None
    return normalized.loc[:, EXPECTED_COLUMNS]


def _ensure_table_schema(conn: sqlite3.Connection, table: str) -> None:
    if not _table_exists(conn, table):
        conn.execute(f"CREATE TABLE {table} ({_table_columns_sql()})")
        return

    existing_columns = set(_get_table_columns(conn, table))
    for col in EXPECTED_COLUMNS:
        if col not in existing_columns:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col} TEXT")


def _ensure_tables(conn: sqlite3.Connection) -> None:
    _ensure_table_schema(conn, TABLE_TENSION_DATA)
    _ensure_table_schema(conn, TABLE_TENSION_SAMPLES)
    conn.commit()


def _table_exists(conn: sqlite3.Connection, table: str) -> bool:
    row = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name=?", (table,)
    ).fetchone()
    return row is not None


def _read_table(conn: sqlite3.Connection, table: str) -> pd.DataFrame:
    if not _table_exists(conn, table):
        return pd.DataFrame(columns=EXPECTED_COLUMNS)
    df = pd.read_sql_query(f"SELECT * FROM {table}", conn)
    return _normalize_dataframe_columns(df)


def _cache_key(file_path: str, table: str) -> str:
    return f"{file_path}::{table}"


def _get_table_dataframe(file_path: str, table: str) -> pd.DataFrame:
    key = _cache_key(file_path, table)
    if key in _dataframe_cache:
        return _dataframe_cache[key]

    if not Path(file_path).exists():
        df = pd.DataFrame(columns=EXPECTED_COLUMNS)
        _dataframe_cache[key] = df
        return df

    with sqlite3.connect(file_path) as conn:
        _ensure_tables(conn)
        df = _read_table(conn, table)
    _dataframe_cache[key] = df
    return df


def _normalize_row(row: dict[str, Any]) -> dict[str, Any]:
    normalized: dict[str, Any] = {}
    for col in EXPECTED_COLUMNS:
        value = row.get(col)
        if isinstance(value, datetime):
            value = value.isoformat()
        elif isinstance(value, list):
            value = str(value)
        normalized[col] = value
    return normalized


def _append_row(file_path: str, table: str, row: dict[str, Any]) -> None:
    normalized = _normalize_row(row)
    columns = ", ".join(EXPECTED_COLUMNS)
    placeholders = ", ".join("?" for _ in EXPECTED_COLUMNS)
    values = [normalized[col] for col in EXPECTED_COLUMNS]

    with sqlite3.connect(file_path) as conn:
        _ensure_tables(conn)
        conn.execute(
            f"INSERT INTO {table} ({columns}) VALUES ({placeholders})",
            values,
        )
        conn.commit()

    key = _cache_key(file_path, table)
    if key in _dataframe_cache:
        cache_df = _dataframe_cache[key].copy()
        cache_df.loc[len(cache_df)] = normalized
        _dataframe_cache[key] = cache_df


def get_dataframe(file_path: str) -> pd.DataFrame:
    """Return the summary measurement DataFrame (``tension_data``)."""

    return _get_table_dataframe(file_path, TABLE_TENSION_DATA)


def update_dataframe(file_path: str, df: pd.DataFrame) -> None:
    """Replace ``tension_data`` with ``df`` and refresh cache."""

    normalized_df = _normalize_dataframe_columns(df)
    key = _cache_key(file_path, TABLE_TENSION_DATA)
    _dataframe_cache[key] = normalized_df.copy()
    with sqlite3.connect(file_path) as conn:
        _ensure_tables(conn)
        normalized_df.to_sql(TABLE_TENSION_DATA, conn, if_exists="replace", index=False)


def append_dataframe_row(file_path: str, row: dict[str, Any]) -> None:
    """Append one row to ``tension_data`` without rewriting the full table."""

    _append_row(file_path, TABLE_TENSION_DATA, row)


def get_results_dataframe(file_path: str) -> pd.DataFrame:
    """Return raw samples from ``tension_samples``.

    For backward compatibility with older databases, this falls back to
    ``tension_data`` when ``tension_samples`` is empty.
    """

    samples = _get_table_dataframe(file_path, TABLE_TENSION_SAMPLES)
    if not samples.empty:
        return samples

    # Backward-compatibility path for historical DBs that stored samples in
    # tension_data only.
    return _get_table_dataframe(file_path, TABLE_TENSION_DATA)


def update_results_dataframe(file_path: str, df: pd.DataFrame) -> None:
    """Replace ``tension_samples`` with ``df`` and refresh cache."""

    normalized_df = _normalize_dataframe_columns(df)
    key = _cache_key(file_path, TABLE_TENSION_SAMPLES)
    _dataframe_cache[key] = normalized_df.copy()
    with sqlite3.connect(file_path) as conn:
        _ensure_tables(conn)
        normalized_df.to_sql(TABLE_TENSION_SAMPLES, conn, if_exists="replace", index=False)


def append_results_row(file_path: str, row: dict[str, Any]) -> None:
    """Append one row to ``tension_samples`` without rewriting the full table."""

    _append_row(file_path, TABLE_TENSION_SAMPLES, row)


def _drop_wire_numbers(
    df: pd.DataFrame,
    apa_name: str,
    layer: str,
    side: str,
    wire_numbers: Iterable[int],
) -> pd.DataFrame:
    """Return ``df`` with the selected wires removed for one APA/layer/side."""

    numbers = {int(wire) for wire in wire_numbers}
    if df.empty or not numbers:
        return df.reset_index(drop=True)

    wire_series = pd.to_numeric(df["wire_number"], errors="coerce")
    mask = ~(
        (df["apa_name"] == apa_name)
        & (df["layer"] == layer)
        & (df["side"] == side)
        & wire_series.isin(numbers)
    )
    return df[mask].reset_index(drop=True)


def clear_wire_numbers(
    file_path: str,
    apa_name: str,
    layer: str,
    side: str,
    wire_numbers: Iterable[int],
) -> None:
    """Remove all rows matching ``wire_numbers`` from both DB tables."""

    numbers = sorted({int(wire) for wire in wire_numbers})
    if not numbers:
        return

    df = get_dataframe(file_path)
    update_dataframe(file_path, _drop_wire_numbers(df, apa_name, layer, side, numbers))

    samples_df = _get_table_dataframe(file_path, TABLE_TENSION_SAMPLES)
    update_results_dataframe(
        file_path,
        _drop_wire_numbers(samples_df, apa_name, layer, side, numbers),
    )


def clear_wire_range(
    file_path: str,
    apa_name: str,
    layer: str,
    side: str,
    start: int,
    end: int,
) -> None:
    """Remove all rows matching the given wire range from both DB tables."""

    clear_wire_numbers(file_path, apa_name, layer, side, range(start, end + 1))


def find_outliers(
    file_path: str,
    apa_name: str,
    layer: str,
    side: str,
    times_sigma: float = 2.5,
    confidence_threshold: float = 0.0,
) -> list[int]:
    """Find wire numbers whose tension residual exceeds ``times_sigma`` std."""

    df = get_dataframe(file_path)
    mask = (
        (df["apa_name"] == apa_name)
        & (df["layer"] == layer)
        & (df["side"] == side)
        & (df["confidence"].astype(float) >= confidence_threshold)
    )
    subset = df[mask].copy()
    subset["tension"] = pd.to_numeric(subset["tension"], errors="coerce")
    subset["wire_number"] = pd.to_numeric(subset["wire_number"], errors="coerce")
    subset = subset.dropna(subset=["tension", "wire_number"])
    if subset.empty:
        return []

    subset = subset.sort_values("wire_number")

    rolling_mean = (
        subset["tension"].rolling(window=20, center=True, min_periods=20).mean()
    )
    residuals = subset["tension"] - rolling_mean
    resid_std = residuals.std(skipna=True)

    if pd.isna(resid_std) or resid_std == 0:
        return []

    is_outlier = rolling_mean.notna() & (residuals.abs() > times_sigma * resid_std)
    outliers = subset.loc[is_outlier, "wire_number"].astype(int).tolist()
    return sorted(set(outliers))


def find_distribution_outliers(
    file_path: str,
    apa_name: str,
    layer: str,
    side: str,
    times_sigma: float = 2.5,
    confidence_threshold: float = 0.0,
) -> list[int]:
    """Find wires whose tension lies far from the bulk tension distribution."""

    df = get_dataframe(file_path)
    mask = (
        (df["apa_name"] == apa_name)
        & (df["layer"] == layer)
        & (df["side"] == side)
        & (df["confidence"].astype(float) >= confidence_threshold)
    )
    subset = df[mask].copy()
    subset["tension"] = pd.to_numeric(subset["tension"], errors="coerce")
    subset["wire_number"] = pd.to_numeric(subset["wire_number"], errors="coerce")
    subset = subset.dropna(subset=["tension", "wire_number"])
    if subset.empty:
        return []

    tension_mean = subset["tension"].mean(skipna=True)
    tension_std = subset["tension"].std(skipna=True)
    if pd.isna(tension_mean) or pd.isna(tension_std) or tension_std == 0:
        return []

    is_outlier = (subset["tension"] - tension_mean).abs() > times_sigma * tension_std
    outliers = subset.loc[is_outlier, "wire_number"].astype(int).tolist()
    return sorted(set(outliers))
