import pandas as pd
from pathlib import Path

# Global cache
_dataframe_cache = {}

# Fixed expected columns
EXPECTED_COLUMNS = [
    "layer",
    "side",
    "wire_number",
    "wire_length",
    "tension",
    "tension_pass",
    "frequency",
    "zone",
    "confidence",
    "t_sigma",
    "x",
    "y",
    "wires",
    "ttf",
    "time",
]


def get_dataframe(file_path: str) -> pd.DataFrame:
    """Get a cached DataFrame or read from disk if not cached."""
    if file_path not in _dataframe_cache:
        if Path(file_path).exists():
            df = pd.read_csv(file_path, skiprows=1, names=EXPECTED_COLUMNS)
        else:
            df = pd.DataFrame(columns=EXPECTED_COLUMNS)
        _dataframe_cache[file_path] = df
    return _dataframe_cache[file_path]


def update_dataframe(file_path: str, df: pd.DataFrame) -> None:
    """Update the cache and write to disk."""
    _dataframe_cache[file_path] = df.copy()
    df.to_csv(file_path, index=False)


def invalidate_cache(file_path: str) -> None:
    """Remove a cached DataFrame (e.g. if file changes externally)."""
    _dataframe_cache.pop(file_path, None)
