import pandas as pd
import sqlite3
from pathlib import Path
from results import EXPECTED_COLUMNS, RAW_SAMPLE_COLUMNS

# Global cache
_dataframe_cache: dict[str, pd.DataFrame] = {}

# Fixed expected columns come from TensionResult dataclass


def _ensure_data_table(conn: sqlite3.Connection) -> None:
    columns_sql = ", ".join(f"{col} TEXT" for col in EXPECTED_COLUMNS)
    conn.execute(f"CREATE TABLE IF NOT EXISTS tension_data ({columns_sql})")
    conn.commit()


def _ensure_samples_table(conn: sqlite3.Connection) -> None:
    columns_sql = ", ".join(f"{col} TEXT" for col in RAW_SAMPLE_COLUMNS)
    conn.execute(f"CREATE TABLE IF NOT EXISTS tension_samples ({columns_sql})")
    conn.commit()


def get_dataframe(file_path: str) -> pd.DataFrame:
    """Get a cached DataFrame or read from disk if not cached."""
    if file_path not in _dataframe_cache:
        if Path(file_path).exists():
            with sqlite3.connect(file_path) as conn:
                _ensure_data_table(conn)
                df = pd.read_sql_query("SELECT * FROM tension_data", conn)
        else:
            df = pd.DataFrame(columns=EXPECTED_COLUMNS)
        _dataframe_cache[file_path] = df
    return _dataframe_cache[file_path]


def update_dataframe(file_path: str, df: pd.DataFrame) -> None:
    """Update the cache and write to disk."""
    _dataframe_cache[file_path] = df.copy()
    with sqlite3.connect(file_path) as conn:
        _ensure_data_table(conn)
        df.to_sql("tension_data", conn, if_exists="replace", index=False)


def get_samples_dataframe(file_path: str) -> pd.DataFrame:
    """Return DataFrame of raw samples."""
    key = f"samples:{file_path}"
    if key not in _dataframe_cache:
        if Path(file_path).exists():
            with sqlite3.connect(file_path) as conn:
                _ensure_samples_table(conn)
                df = pd.read_sql_query("SELECT * FROM tension_samples", conn)
        else:
            df = pd.DataFrame(columns=RAW_SAMPLE_COLUMNS)
        _dataframe_cache[key] = df
    return _dataframe_cache[key]


def update_samples_dataframe(file_path: str, df: pd.DataFrame) -> None:
    """Write raw sample DataFrame to disk and update cache."""
    key = f"samples:{file_path}"
    _dataframe_cache[key] = df.copy()
    with sqlite3.connect(file_path) as conn:
        _ensure_samples_table(conn)
        df.to_sql("tension_samples", conn, if_exists="replace", index=False)


def clear_wire_range(
    file_path: str,
    apa_name: str,
    layer: str,
    side: str,
    start: int,
    end: int,
) -> None:
    """Remove all rows matching the given wire range from the database."""

    # Clear summarized tension data
    df = get_dataframe(file_path)
    mask = ~(
        (df["apa_name"] == apa_name)
        & (df["layer"] == layer)
        & (df["side"] == side)
        & (df["wire_number"].astype(int).between(start, end))
    )
    update_dataframe(file_path, df[mask].reset_index(drop=True))

    # Clear raw sample data
    samples_df = get_samples_dataframe(file_path)
    mask_s = ~(
        (samples_df["apa_name"] == apa_name)
        & (samples_df["layer"] == layer)
        & (samples_df["side"] == side)
        & (samples_df["wire_number"].astype(int).between(start, end))
    )
    update_samples_dataframe(file_path, samples_df[mask_s].reset_index(drop=True))


def clear_outliers(
    file_path: str,
    apa_name: str,
    layer: str,
    side: str,
    sigma: float = 3.0,
    confidence_threshold: float = 0.0,
) -> list[int]:
    """Remove wires with tension more than ``sigma`` standard deviations from the mean.

    Returns the list of wire numbers removed.
    """
    df = get_dataframe(file_path)
    mask = (
        (df["apa_name"] == apa_name)
        & (df["layer"] == layer)
        & (df["side"] == side)
        & (df["confidence"].astype(float) >= confidence_threshold)
    )
    subset = df[mask].copy()
    subset["tension"] = pd.to_numeric(subset["tension"], errors="coerce")
    subset = subset.dropna(subset=["tension", "wire_number"])
    if subset.empty:
        return []
    mean = subset["tension"].mean()
    std = subset["tension"].std()
    if pd.isna(std) or std == 0:
        return []
    outliers = (
        subset[(subset["tension"] - mean).abs() > sigma * std]["wire_number"]
        .astype(int)
        .tolist()
    )
    for wire in outliers:
        clear_wire_range(file_path, apa_name, layer, side, wire, wire)
    return outliers
