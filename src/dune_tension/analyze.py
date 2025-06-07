import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from data_cache import get_dataframe, EXPECTED_COLUMNS
import os
from typing import Dict, List, Tuple, Any
from tensiometer_functions import TensiometerConfig
from tension_calculation import (
    has_cluster_dict,
    tension_plausible,
    calculate_kde_max,
)
from tensiometer import TensionResult


def greedy_wire_ordering_with_bounds_tiebreak(existing_wires, expected_range):
    expected = list(expected_range)
    missing = sorted(set(expected) - set(existing_wires))

    if not missing:
        return []

    bounds = [min(expected), max(expected)]
    closest_to_bounds = min(
        existing_wires, key=lambda w: min(abs(w - b) for b in bounds)
    )

    remaining = set(missing)
    result = []
    current = closest_to_bounds

    while remaining:
        # Among remaining wires, find those with minimum distance to current
        min_dist = min(abs(x - current) for x in remaining)
        candidates = [x for x in remaining if abs(x - current) == min_dist]

        # Tie-break by distance to bounds
        next_wire = min(candidates, key=lambda x: min(abs(x - b) for b in bounds))

        result.append(next_wire)
        remaining.remove(next_wire)
        current = next_wire

    return result


def _load_and_analyze(config: TensiometerConfig) -> Dict[str, Any]:
    """Helper that loads the data file and performs analysis."""
    expected_range = get_expected_range(config.layer)
    raw_df = preprocess_dataframe(get_dataframe(config.data_path))
    agg_df = aggregate_samples(raw_df, config)
    df_sorted = agg_df.sort_values(by="time")
    return analyze_by_side(df_sorted, expected_range, config.layer)


def analyze_tension_data(config: TensiometerConfig) -> Dict[str, Any]:
    """Return analysis results and update all output files."""
    results = _load_and_analyze(config)

    log_paths = update_tension_logs(config, _results=results)

    return {
        **log_paths,
        "badwires": results["badwires"],
        "missing_wires": results["missing_wires"],
    }


def get_missing_wires(config: TensiometerConfig) -> Dict[str, List[int]]:
    """Return a dictionary of missing wires for each side."""
    results = _load_and_analyze(config)
    return results["missing_wires"]


def update_tension_logs(
    config: TensiometerConfig, _results: Dict[str, Any] | None = None
) -> Dict[str, str]:
    """Update plot, summaries and bad wire logs for the given configuration."""
    output_dir = "data/tension_plots"
    badwires_path = f"data/badwires/badwires_{config.apa_name}_{config.layer}.txt"
    tension_summary_csv_path = (
        f"data/tension_summaries/tension_summary_{config.apa_name}_{config.layer}.csv"
    )

    os.makedirs(output_dir, exist_ok=True)

    results = _results if _results is not None else _load_and_analyze(config)

    write_summary_csv(results["tension_series"], tension_summary_csv_path)
    save_plot(
        results["line_data"],
        results["hist_data"],
        config.apa_name,
        config.layer,
        output_dir,
    )
    write_badwires(
        badwires_path,
        config.apa_name,
        config.layer,
        results["badwires_by_group"],
        results["outlier_wires_by_group"],
    )

    return {
        "badwires": badwires_path,
        "tension_summary_csv": tension_summary_csv_path,
        "plot_image": f"{output_dir}/tension_plot_{config.apa_name}_{config.layer}.png",
    }


def get_expected_range(layer: str) -> range:
    ranges = {
        "U": range(8, 1147),
        "V": range(8, 1147),
        "X": range(1, 481),
        "G": range(1, 482),
    }
    return ranges.get(layer, range(0))


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    df["wire_number"] = pd.to_numeric(df["wire_number"], errors="coerce")
    df["tension"] = pd.to_numeric(df["tension"], errors="coerce")
    df["tension_pass"] = df["tension_pass"].astype(str) == "True"
    df = df.dropna(subset=["wire_number", "tension"])
    df = df[df["tension"] > 0]
    return df


def aggregate_samples(df: pd.DataFrame, config: TensiometerConfig) -> pd.DataFrame:
    """Aggregate raw samples into one result per wire."""
    rows = []
    grouped = df.groupby(["layer", "side", "wire_number"])
    for (layer, side, wire_number), group in grouped:
        candidates = group[
            (group["confidence"] >= config.confidence_threshold)
            & group["tension"].apply(tension_plausible)
        ]
        samples = [
            TensionResult(
                layer=layer,
                side=side,
                wire_number=int(wire_number),
                frequency=row["frequency"],
                confidence=row["confidence"],
                x=row["x"],
                y=row["y"],
                wires=[row["tension"]],
            )
            for _, row in candidates.iterrows()
        ]
        cluster = has_cluster_dict(samples, "tension", config.samples_per_wire)
        if not cluster:
            continue

        if config.samples_per_wire == 1:
            freq = cluster[0].frequency
            conf = cluster[0].confidence
            x_val = cluster[0].x
            y_val = cluster[0].y
        else:
            freq = calculate_kde_max([s.frequency for s in cluster])
            conf = np.average([s.confidence for s in cluster])
            x_val = round(np.average([s.x for s in cluster]), 1)
            y_val = round(np.average([s.y for s in cluster]), 1)
        tensions = [float(s.tension) for s in cluster]
        res = TensionResult(
            layer=layer,
            side=side,
            wire_number=int(wire_number),
            frequency=freq,
            confidence=conf,
            x=x_val,
            y=y_val,
            wires=tensions,
        )
        res.ttf = float(group["ttf"].max()) if "ttf" in group.columns else 0.0
        res.time = float(group["time"].max()) if "time" in group.columns else None
        rows.append({col: getattr(res, col, None) for col in EXPECTED_COLUMNS})

    return pd.DataFrame(rows, columns=EXPECTED_COLUMNS)


def analyze_by_side(
    df_sorted: pd.DataFrame, expected_range: range, layer: str, k: float = 2.0
) -> Dict[str, Any]:
    badwires_by_group: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    outlier_wires_by_group: Dict[Tuple[str, str], List[int]] = defaultdict(list)
    tension_series: Dict[str, Dict[int, float]] = {"A": {}, "B": {}}
    missing_wires: Dict[str, List[int]] = {"A": [], "B": []}
    line_data: List[pd.DataFrame] = []
    hist_data: List[pd.DataFrame] = []

    latest_df = df_sorted.drop_duplicates(
        subset=["layer", "side", "wire_number"], keep="last"
    )
    grouped_by_side = latest_df.groupby("side")

    expected_set = set(expected_range)
    all_badwires: set[int] = set()

    # Handle sides with no measurements at all
    present_sides = set(str(s) for s in latest_df.get("side", []))
    for side in ["A", "B"]:
        if side not in present_sides:
            missing_wires[side] = sorted(expected_set)
            badwires_by_group[(layer, side)] = sorted(expected_set)
            line_data.append(
                pd.DataFrame(columns=["wire_number", "tension", "side_label"])
            )
            hist_data.append(pd.DataFrame(columns=["tension", "side_label"]))
            all_badwires.update(expected_set)

    for side, group in grouped_by_side:
        group_sorted = group.sort_values(by="wire_number")
        wire_numbers = group_sorted["wire_number"].astype(int).values
        if len(wire_numbers) == 0:
            continue

        # Standard deviation-based outlier detection
        tension_values = group_sorted["tension"]
        mean_tension = tension_values.mean()
        std_tension = tension_values.std()

        outlier_mask = (tension_values < mean_tension - k * std_tension) | (
            tension_values > mean_tension + k * std_tension
        )

        outliers = group_sorted.loc[outlier_mask, "wire_number"].astype(int).tolist()
        outlier_wires_by_group[(layer, side)] = outliers

        expected_set = set(expected_range)
        existing_set = set(wire_numbers)
        group_all = df_sorted[df_sorted["side"] == side]
        tension_ok = group_all.groupby("wire_number")["tension_pass"].any()
        failed = set(tension_ok[~tension_ok].index.astype(int))

        missing = (
            greedy_wire_ordering_with_bounds_tiebreak(
                list(existing_set), list(expected_set)
            )
            if existing_set
            else list(sorted(expected_set))
        )
        missing_wires[side] = missing

        badwires = sorted((expected_set - existing_set) | (expected_set & failed))
        badwires_by_group[(layer, side)] = badwires
        all_badwires.update(badwires)

        for _, row in group_sorted.iterrows():
            tension_series[side][int(row["wire_number"])] = row["tension"]

        group_sorted["side_label"] = f"Side {side}"
        line_data.append(group_sorted[["wire_number", "tension", "side_label"]])
        hist_data.append(group_sorted[["tension", "side_label"]])

    return {
        "badwires_by_group": badwires_by_group,
        "outlier_wires_by_group": outlier_wires_by_group,
        "tension_series": tension_series,
        "missing_wires": missing_wires,
        "line_data": line_data,
        "hist_data": hist_data,
        "badwires": sorted(all_badwires),
    }


def write_summary_csv(tension_series: Dict[str, Dict[int, float]], path: str) -> None:
    all_wires = sorted(
        set(tension_series["A"].keys()) | set(tension_series["B"].keys())
    )
    summary_df = pd.DataFrame(
        {
            "wire_number": all_wires,
            "A": [tension_series["A"].get(w, np.nan) for w in all_wires],
            "B": [tension_series["B"].get(w, np.nan) for w in all_wires],
        }
    )
    summary_df.to_csv(path, index=False)


def save_plot(
    line_data: List[pd.DataFrame],
    hist_data: List[pd.DataFrame],
    apa_name: str,
    layer: str,
    output_dir: str,
) -> None:
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
        f"{output_dir}/tension_plot_{apa_name}_layer_{layer}_combined.png", dpi=300
    )
    plt.close()


def write_badwires(
    path: str,
    apa_name: str,
    layer: str,
    badwires_by_group: Dict[Tuple[str, str], List[int]],
    outlier_wires_by_group: Dict[Tuple[str, str], List[int]],
) -> None:
    with open(path, "w") as f:
        for (layer_val, side), badwires in badwires_by_group.items():
            f.write(f"{apa_name} - Layer {layer_val}, Side {side}:\n")
            if badwires:
                f.write(
                    "  Bad wire_numbers (missing or no tension_pass=True): "
                    + ", ".join(map(str, badwires))
                    + "\n"
                )
            else:
                f.write("  No bad wire_numbers\n")

            outliers = sorted(set(outlier_wires_by_group.get((layer_val, side), [])))
            if outliers:
                f.write(
                    "  Outlier wire_numbers (far from moving average): "
                    + ", ".join(map(str, outliers))
                    + "\n"
                )
            else:
                f.write("  No outlier wire_numbers\n")
            f.write("\n")


if __name__ == "__main__":
    tasks = [
        # ("US_APA7", "U"),
        # ("US_APA7", "V"),
        # ("US_APA7", "X"),
        ("US_APA9", "X")
    ]

    for apa_name, layer in tasks:
        print(f"Processing APA {apa_name}, Layer {layer}...")
        try:
            config = TensiometerConfig(
                apa_name=apa_name,
                layer=layer,
            )  # type: ignore
            results = analyze_tension_data(config)
            print("  Plot:", results["plot_image"])
            print("  Summary CSV:", results["tension_summary_csv"])
            print("  Bad Wires Log:", results["badwires"])
        except FileNotFoundError:
            print(f"  ❌ File not found for {apa_name}, Layer {layer}. Skipping.")
        except Exception as e:
            print(f"  ❌ Error processing {apa_name}, Layer {layer}: {e}")
        print()
