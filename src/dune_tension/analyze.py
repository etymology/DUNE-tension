import os
from datetime import datetime
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from data_cache import get_samples_dataframe
from results import TensionResult
from tensiometer_functions import TensiometerConfig
from tension_calculation import calculate_kde_max, has_cluster


def get_expected_range(layer: str) -> range:
    """Return the expected wire range for a given layer."""
    ranges = {
        "U": range(8, 1147),
        "V": range(8, 1147),
        "X": range(1, 481),
        "G": range(1, 482),
    }
    return ranges.get(layer, range(0))


def _compute_tensions(
    config: TensiometerConfig, samples: pd.DataFrame
) -> tuple[
    Dict[str, Dict[int, float]],
    List[pd.DataFrame],
    List[pd.DataFrame],
    Dict[str, List[int]],
]:
    """Return tension series and plotting DataFrames grouped by side."""
    tension_series: Dict[str, Dict[int, float]] = {"A": {}, "B": {}}
    line_data: List[pd.DataFrame] = []
    hist_data: List[pd.DataFrame] = []
    missing: Dict[str, List[int]] = {"A": [], "B": []}

    expected = get_expected_range(config.layer)
    for side in ["A", "B"]:
        side_df = samples[samples["side"] == side]
        measured: Dict[int, float] = {}
        for wire in expected:
            wire_df = side_df[side_df["wire_number"] == wire]
            if len(wire_df) < config.samples_per_wire:
                continue
            records = [
                TensionResult(
                    apa_name=row.apa_name,
                    layer=row.layer,
                    side=row.side,
                    wire_number=int(row.wire_number),
                    frequency=float(row.frequency),
                    confidence=float(row.confidence),
                    x=float(row.x),
                    y=float(row.y),
                    wires=[],
                    time=datetime.fromisoformat(row.time)
                    if isinstance(row.time, str)
                    else row.time,
                )
                for row in wire_df.itertuples()
            ]
            cluster = has_cluster(records, "tension", config.samples_per_wire)
            if not cluster:
                continue
            freq = calculate_kde_max([r.frequency for r in cluster])
            conf = float(np.average([r.confidence for r in cluster]))
            x = round(float(np.average([r.x for r in cluster])), 1)
            y = round(float(np.average([r.y for r in cluster])), 1)
            tr = TensionResult(
                apa_name=config.apa_name,
                layer=config.layer,
                side=side,
                wire_number=wire,
                frequency=freq,
                confidence=conf,
                x=x,
                y=y,
                wires=[float(r.tension) for r in cluster],
                time=datetime.now(),
            )
            measured[wire] = tr.tension
            line_data.append(
                pd.DataFrame(
                    {
                        "wire_number": [wire],
                        "tension": [tr.tension],
                        "side_label": f"Side {side}",
                    }
                )
            )
            hist_data.append(
                pd.DataFrame({"tension": [tr.tension], "side_label": f"Side {side}"})
            )
        tension_series[side] = measured
        missing[side] = sorted(set(expected) - set(measured.keys()))
    return tension_series, line_data, hist_data, missing


def write_summary_csv(tension_series: Dict[str, Dict[int, float]], path: str) -> None:
    """Write tension summary CSV for both sides."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    all_wires = sorted(
        set(tension_series["A"].keys()) | set(tension_series["B"].keys())
    )
    df = pd.DataFrame(
        {
            "wire_number": all_wires,
            "A": [tension_series["A"].get(w, np.nan) for w in all_wires],
            "B": [tension_series["B"].get(w, np.nan) for w in all_wires],
        }
    )
    df.to_csv(path, index=False)


def save_plot(
    line_data: List[pd.DataFrame],
    hist_data: List[pd.DataFrame],
    apa_name: str,
    layer: str,
    output_dir: str,
) -> None:
    os.makedirs(output_dir, exist_ok=True)
    if not line_data:
        return
    line_df = pd.concat(line_data)
    hist_df = pd.concat(hist_data)

    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    for side_label, group in line_df.groupby("side_label"):
        plt.scatter(
            group["wire_number"], group["tension"], label=side_label, alpha=0.5, s=10
        )
        sorted_group = group.sort_values("wire_number")
        ma = sorted_group["tension"].rolling(window=15, center=True).mean()
        plt.plot(sorted_group["wire_number"], ma, alpha=0.4, linewidth=2)
    plt.title(f"{apa_name} - Tension Scatter Plot with Trendline - Layer {layer}")
    plt.xlabel("Wire Number")
    plt.ylabel("Tension")
    plt.grid(True, linestyle=":", linewidth=0.5, color="gray")
    plt.legend()

    plt.subplot(1, 2, 2)
    sns.histplot(
        data=hist_df,
        x="tension",
        hue="side_label",
        element="step",
        stat="count",
        common_norm=False,
    )
    plt.title(f"{apa_name} - Tension Histogram - Layer {layer}")
    plt.xlabel("Tension")
    plt.ylabel("Count")
    plt.grid(True, linestyle=":", linewidth=0.5, color="gray")

    plt.tight_layout()
    plt.savefig(
        os.path.join(output_dir, f"tension_plot_{apa_name}_{layer}.png"), dpi=300
    )
    plt.close()


def write_missing_wires(
    path: str, apa: str, layer: str, missing: Dict[str, List[int]]
) -> None:
    """Write a simple log of wires with no measurements."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w") as f:
        for side in ["A", "B"]:
            wires = missing.get(side, [])
            f.write(f"{apa} - Layer {layer}, Side {side}:\n")
            if wires:
                f.write("  Missing wire_numbers: " + ", ".join(map(str, wires)) + "\n")
            else:
                f.write("  All wires measured\n")
        f.write("\n")


def _order_missing_wires(missing: List[int], measured: List[int]) -> List[int]:
    """Return ``missing`` ordered close to ``measured`` then by proximity."""
    if not missing:
        return []

    if measured:
        start = min(
            missing,
            key=lambda w: (min(abs(w - m) for m in measured), w),
        )
    else:
        start = min(missing)

    remaining = set(missing)
    ordered = [start]
    remaining.remove(start)
    current = start

    while remaining:
        next_wire = min(remaining, key=lambda w: (abs(w - current), w))
        ordered.append(next_wire)
        remaining.remove(next_wire)
        current = next_wire

    return ordered


def update_tension_logs(config: TensiometerConfig) -> Dict[str, str]:
    """Generate summary CSV, plot and missing wire log for ``config``."""
    samples = get_samples_dataframe(config.data_path)
    mask = (
        (samples["apa_name"] == config.apa_name)
        & (samples["layer"] == config.layer)
        & (samples["confidence"].astype(float) >= config.confidence_threshold)
    )
    df = samples[mask].copy()
    df["wire_number"] = pd.to_numeric(df["wire_number"], errors="coerce")
    df = df.dropna(subset=["wire_number", "frequency"])
    df["wire_number"] = df["wire_number"].astype(int)

    tension_series, line_data, hist_data, missing = _compute_tensions(config, df)

    output_dir = "data/tension_plots"
    summary_path = (
        f"data/tension_summaries/tension_summary_{config.apa_name}_{config.layer}.csv"
    )
    bad_path = f"data/badwires/badwires_{config.apa_name}_{config.layer}.txt"

    write_summary_csv(tension_series, summary_path)
    save_plot(line_data, hist_data, config.apa_name, config.layer, output_dir)
    write_missing_wires(bad_path, config.apa_name, config.layer, missing)

    return {
        "badwires": bad_path,
        "tension_summary_csv": summary_path,
        "plot_image": os.path.join(
            output_dir, f"tension_plot_{config.apa_name}_{config.layer}.png"
        ),
    }


def get_missing_wires(config: TensiometerConfig) -> Dict[str, List[int]]:
    """Return missing wire numbers for each side for ``config``."""
    samples = get_samples_dataframe(config.data_path)
    mask = (
        (samples["apa_name"] == config.apa_name)
        & (samples["layer"] == config.layer)
        & (samples["confidence"].astype(float) >= config.confidence_threshold)
    )
    df = samples[mask].copy()
    df["wire_number"] = pd.to_numeric(df["wire_number"], errors="coerce")
    df = df.dropna(subset=["wire_number", "frequency"])
    df["wire_number"] = df["wire_number"].astype(int)

    tension_series, _, _, missing = _compute_tensions(config, df)

    for side in ["A", "B"]:
        missing_side = missing.get(side, [])
        measured = list(tension_series.get(side, {}).keys())
        missing[side] = _order_missing_wires(missing_side, measured)

    return missing
