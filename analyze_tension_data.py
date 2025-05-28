import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from collections import defaultdict
from itertools import groupby
from operator import itemgetter
import os


def prioritize_missing_wires_by_proximity(
    missing_wires: list[int], present_wires: list[int]
) -> list[int]:
    """Sort missing wires by proximity to existing wires (minimum index distance)."""
    from bisect import bisect_left

    present_sorted = sorted(present_wires)
    prioritized = sorted(
        missing_wires,
        key=lambda x: min(
            abs(x - present_sorted[bisect_left(present_sorted, x) - 1])
            if bisect_left(present_sorted, x) > 0
            else float("inf"),
            abs(x - present_sorted[bisect_left(present_sorted, x)])
            if bisect_left(present_sorted, x) < len(present_sorted)
            else float("inf"),
        ),
    )
    return prioritized


def compress_ranges(numbers):
    """Convert sorted list of integers into range strings like 'x-y' or single 'x'."""
    ranges = []
    for k, g in groupby(enumerate(numbers), lambda ix: ix[1] - ix[0]):
        group = list(map(itemgetter(1), g))
        if len(group) == 1:
            ranges.append(f"{group[0]}")
        else:
            ranges.append(f"{group[0]}-{group[-1]}")
    return ranges


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


def analyze_tension_data(apa_name, layer):
    input_csv = f"data/tension_data/tension_data_{apa_name}_{layer}.csv"
    output_dir = f"data/tension_plots_{apa_name}"
    bad_wires_log_path = f"data/bad_wires/bad_wires_log_{apa_name}_{layer}.txt"
    tension_summary_csv_path = (
        f"data/tension_summaries/tension_summary_{apa_name}_{layer}.csv"
    )

    expected_columns = [
        "layer",
        "side",
        "wire_number",
        "tension",
        "tension_pass",
        "frequency",
        "zone",
        "confidence",
        "t_sigma",
        "x",
        "y",
        "Gcode",
        "wires",
        "ttf",
        "time",
    ]

    expected_wire_ranges = {
        "U": range(8, 1147),
        "V": range(8, 1147),
        "X": range(1, 481),
        "G": range(1, 482),
    }
    expected_range = expected_wire_ranges.get(layer, [])

    os.makedirs(output_dir, exist_ok=True)

    try:
        df = pd.read_csv(input_csv, skiprows=1, names=expected_columns)
    except FileNotFoundError:
        print(f"File not found: {input_csv}")
        return {"error": f"File not found: {input_csv}"}
    df["wire_number"] = pd.to_numeric(df["wire_number"], errors="coerce")
    df["tension"] = pd.to_numeric(df["tension"], errors="coerce")
    df["tension_pass"] = df["tension_pass"].astype(str) == "True"
    df = df.dropna(subset=["wire_number", "tension"])
    df = df[df["tension"] > 0]
    df_sorted = df.sort_values(by="time")

    bad_wires_by_group = defaultdict(list)
    outlier_wires_by_group = defaultdict(list)
    all_passed_df = df_sorted[df_sorted["tension_pass"]]

    latest_df = df_sorted.drop_duplicates(
        subset=["layer", "side", "wire_number"], keep="last"
    )
    grouped_by_side = latest_df.groupby("side")

    line_data = []
    hist_data = []
    tension_series = {"A": {}, "B": {}}

    for side, group in grouped_by_side:
        group_sorted = group.sort_values(by="wire_number")
        wire_numbers = group_sorted["wire_number"].astype(int).values

        if len(wire_numbers) == 0:
            continue

        # Detect outliers: tension values far (e.g., >20%) from moving average
        sorted_group = group_sorted.sort_values(by="wire_number")
        ma_series = sorted_group["tension"].rolling(window=15, center=True).mean()
        deviation = (sorted_group["tension"] - ma_series).abs()
        outlier_mask = deviation > (0.1 * ma_series)
        outliers = sorted_group.loc[outlier_mask, "wire_number"].astype(int).tolist()
        outlier_wires_by_group[(layer, side)] += outliers
        expected_set = set(expected_range)
        existing_set = set(wire_numbers)
        group_all = df_sorted[(df_sorted["side"] == side)]
        tension_ok = group_all.groupby("wire_number")["tension_pass"].any()
        failed = set(tension_ok[~tension_ok].index.astype(int))

        missing_wires = greedy_wire_ordering_with_bounds_tiebreak(
            list(existing_set), list(expected_set)
        )

        bad_wires = sorted((expected_set - existing_set) | (expected_set & failed))
        bad_wires_by_group[(layer, side)] = bad_wires

        for _, row in group_sorted.iterrows():
            tension_series[side][int(row["wire_number"])] = row["tension"]

        group_sorted["side_label"] = f"Side {side}"
        line_data.append(group_sorted[["wire_number", "tension", "side_label"]])
        hist_data.append(group_sorted[["tension", "side_label"]])

    all_wires = sorted(
        set(tension_series["A"].keys()) | set(tension_series["B"].keys())
    )
    summary_df = pd.DataFrame(
        {
            "wire_number": all_wires,
            "A": [tension_series["A"].get(wire, np.nan) for wire in all_wires],
            "B": [tension_series["B"].get(wire, np.nan) for wire in all_wires],
        }
    )
    summary_df.to_csv(tension_summary_csv_path, index=False)

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
    filename = f"{output_dir}/tension_plot_{apa_name}_layer_{layer}_combined.png"
    plt.savefig(filename, dpi=300)
    plt.close()

    with open(bad_wires_log_path, "w") as f:
        for (layer_val, side), bad_wires in bad_wires_by_group.items():
            f.write(f"{apa_name} - Layer {layer_val}, Side {side}:\n")
            if bad_wires:
                compressed = compress_ranges(bad_wires)
                f.write(
                    "  Bad wire_numbers (missing or no tension_pass=True): "
                    + ", ".join(compressed)
                    + "\n"
                )
            else:
                f.write("  No bad wire_numbers\n")

            outliers = sorted(set(outlier_wires_by_group.get((layer_val, side), [])))
            if outliers:
                compressed_outliers = compress_ranges(outliers)
                f.write(
                    "  Outlier wire_numbers (far from moving average): "
                    + ", ".join(compressed_outliers)
                    + "\n"
                )
            else:
                f.write("  No outlier wire_numbers\n")
            f.write("\n")

    return {
        "bad_wires_log": bad_wires_log_path,
        "tension_summary_csv": tension_summary_csv_path,
        "plot_image": filename,
        "bad_wires": bad_wires,
        "missing_wires": missing_wires,
    }


if __name__ == "__main__":
    tasks = [
        # ("US_APA7", "U"),
        # ("US_APA7", "V"),
        # ("US_APA7", "X"),
        ("US_APA7", "G")
    ]

    for apa_name, layer in tasks:
        print(f"Processing APA {apa_name}, Layer {layer}...")
        try:
            results = analyze_tension_data(apa_name, layer)
            print("  Plot:", results["plot_image"])
            print("  Summary CSV:", results["tension_summary_csv"])
            print("  Bad Wires Log:", results["bad_wires_log"])
        except FileNotFoundError:
            print(f"  ❌ File not found for {apa_name}, Layer {layer}. Skipping.")
        except Exception as e:
            print(f"  ❌ Error processing {apa_name}, Layer {layer}: {e}")
        print()
