import os
from datetime import datetime
from glob import glob

import pandas as pd

from results import TensionResult, EXPECTED_COLUMNS
from data_cache import get_dataframe, update_dataframe


def parse_time(value: str) -> datetime:
    """Parse timestamps from old CSV files."""
    for fmt in ("%Y-%m-%d_%H-%M-%S", "%Y-%m-%d %H:%M:%S"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(value)
    except ValueError:
        return datetime.now()


def parse_wires(val: str) -> list[float]:
    if not isinstance(val, str) or val == "" or val == "nan":
        return []
    try:
        return [float(x) for x in eval(val)]
    except Exception:
        try:
            return [float(val)]
        except Exception:
            return []


def migrate_csvs(csv_dir: str = "data/tension_data", db_path: str | None = None) -> None:
    """Migrate all ``tension_data_*.csv`` files into a single SQLite DB."""
    if db_path is None:
        db_path = os.path.join(csv_dir, "tension_data.db")

    rows = []
    for csv_path in glob(os.path.join(csv_dir, "tension_data_*.csv")):
        base = os.path.basename(csv_path)
        name = base.replace("tension_data_", "").replace(".csv", "")
        if "_" not in name:
            continue
        apa_name, layer = name.split("_", 1)
        try:
            df = pd.read_csv(csv_path)
        except Exception:
            # Fallback for legacy files that may have malformed rows or
            # inconsistent column counts.  The Python CSV engine is more
            # permissive and ``on_bad_lines='skip'`` will drop unreadable lines.
            df = pd.read_csv(csv_path, engine="python", on_bad_lines="skip")

        # Some legacy CSV files are missing newer columns such as ``wires`` or
        # ``ttf``. Ensure all expected keys exist so row.get() works uniformly
        # below.  Use sensible defaults when a column is absent.
        missing_defaults = {
            "side": "",
            "wire_number": 0,
            "frequency": 0.0,
            "confidence": 0.0,
            "x": 0.0,
            "y": 0.0,
            "wires": "",
            "ttf": 0.0,
            "time": "",
        }
        for col, default in missing_defaults.items():
            if col not in df.columns:
                df[col] = default
        for _, row in df.iterrows():
            time_val = parse_time(str(row.get("time", "")))
            wires = parse_wires(row.get("wires", ""))
            # ``ttf`` might be missing or contain garbage from malformed CSV rows
            ttf_val = row.get("ttf", 0.0)
            try:
                ttf = float(ttf_val)
            except Exception:
                ttf = 0.0

            tr = TensionResult(
                apa_name=apa_name,
                layer=str(row.get("layer", layer)),
                side=str(row.get("side", "")),
                wire_number=int(row.get("wire_number", 0)),
                frequency=float(row.get("frequency", 0.0)),
                confidence=float(row.get("confidence", 0.0)),
                x=float(row.get("x", 0.0)),
                y=float(row.get("y", 0.0)),
                wires=wires,
                ttf=ttf,
                time=time_val,
            )
            item = {col: getattr(tr, col) for col in EXPECTED_COLUMNS}
            item["time"] = item["time"].isoformat()
            item["wires"] = str(item["wires"])
            rows.append(item)

    if not rows:
        print("No CSV files found to migrate.")
        return

    df_new = pd.DataFrame(rows, columns=EXPECTED_COLUMNS)
    if os.path.exists(db_path):
        df_existing = get_dataframe(db_path)
        df_all = pd.concat([df_existing, df_new], ignore_index=True)
    else:
        df_all = df_new
    update_dataframe(db_path, df_all)
    print(f"Migrated {len(rows)} rows into {db_path}")


if __name__ == "__main__":
    migrate_csvs()
