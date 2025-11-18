"""Plot tension history for specific wires using raw sample data."""

from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def _load_samples(
    database_path: Path,
    apa_name: str,
    layer: str,
    side: str,
    wire_numbers: list[int],
) -> pd.DataFrame:
    if not wire_numbers:
        message = "wire_numbers must contain at least one wire"
        raise ValueError(message)
    query = """
        SELECT apa_name, layer, side, wire_number, tension, time
        FROM tension_samples
        WHERE apa_name = ?
          AND layer = ?
          AND side = ?
          AND wire_number IN ({placeholders})
        ORDER BY wire_number, time
    """
    placeholders = ", ".join("?" for _ in wire_numbers)
    sql = query.format(placeholders=placeholders)
    with sqlite3.connect(database_path) as connection:
        frame = pd.read_sql_query(
            sql,
            connection,
            params=[apa_name, layer, side, *wire_numbers],
        )
    frame["wire_number"] = frame["wire_number"].astype(int)
    frame["tension"] = pd.to_numeric(frame["tension"], errors="coerce")
    frame["time"] = pd.to_datetime(frame["time"], errors="coerce")
    frame = frame.dropna(subset=["tension", "time"])
    return frame


def plot_wire_tensions(
    database_path: Path,
    apa_name: str,
    layer: str,
    side: str,
    wire_numbers: list[int],
) -> None:
    samples = _load_samples(database_path, apa_name, layer, side, wire_numbers)

    plt.figure(figsize=(12, 6))
    plotted_any = False
    for wire_number in wire_numbers:
        wire_samples = samples[samples["wire_number"] == wire_number]
        if wire_samples.empty:
            continue

        wire_samples = wire_samples.sort_values("time")
        plt.step(
            wire_samples["time"],
            wire_samples["tension"],
            where="post",
            label=f"Wire {wire_number}",
        )
        plt.scatter(
            wire_samples["time"],
            wire_samples["tension"],
            s=20,
            alpha=0.7,
        )
        plotted_any = True

    plt.title(f"Tension over time for APA {apa_name}, Layer {layer}, Side {side}")
    plt.xlabel("Time")
    plt.ylabel("Tension")
    plt.grid(True, linestyle=":", linewidth=0.5, color="gray")
    if plotted_any:
        plt.legend()
        plt.gcf().autofmt_xdate()
    plt.tight_layout()
    plt.show()


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=("Plot the tension history of selected wires using stored samples.")
    )
    parser.add_argument("apa_name", help="APA name to query")
    parser.add_argument("layer", help="Layer identifier (e.g. U, V, X, G)")
    parser.add_argument("side", help="Side identifier (A or B)")
    parser.add_argument(
        "wire_numbers",
        nargs="+",
        type=int,
        help="Wire numbers to include in the plot",
    )
    parser.add_argument(
        "--database",
        type=Path,
        default=Path("tension_data.db"),
        help="Path to the tension sample SQLite database.",
    )
    return parser


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    database_path: Path = args.database
    if not database_path.exists():
        message = f"Database not found at {database_path}"
        raise FileNotFoundError(message)

    plot_wire_tensions(
        database_path=database_path,
        apa_name=args.apa_name,
        layer=args.layer,
        side=args.side,
        wire_numbers=args.wire_numbers,
    )


if __name__ == "__main__":
    main()
