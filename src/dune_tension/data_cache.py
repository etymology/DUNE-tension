import pandas as pd
import sqlite3
from pathlib import Path
from results import RAW_SAMPLE_COLUMNS

# Global cache for dataframes
_dataframe_cache: dict[str, pd.DataFrame] = {}


def _ensure_samples_table(conn: sqlite3.Connection) -> None:
    columns_sql = ", ".join(f"{col} TEXT" for col in RAW_SAMPLE_COLUMNS)
    conn.execute(
        f"CREATE TABLE IF NOT EXISTS tension_samples ({columns_sql})"
    )
    conn.commit()


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
