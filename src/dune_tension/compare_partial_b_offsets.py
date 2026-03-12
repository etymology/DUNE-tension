from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dune_tension.compare_tension_sources import load_action_json, load_summary_csv


def _rolling_mean(series: pd.Series, window: int = 15) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def _series_label(name: str, values: pd.Series) -> str:
    return (
        f"{name} "
        f"(mean={values.mean():.3f}, std={values.std(ddof=0):.3f}, n={int(values.size)})"
    )


def _pair_stats(chicago: pd.Series, uk: pd.Series) -> dict[str, float]:
    diff = chicago - uk
    return {
        "corr": float(chicago.corr(uk)),
        "mean": float(diff.mean()),
        "median": float(diff.median()),
        "std": float(diff.std(ddof=0)),
        "mae": float(diff.abs().mean()),
        "rmse": float(np.sqrt(np.mean(diff**2))),
        "negative_frac": float((diff < 0).mean()),
    }


def _pair_stats_text(chicago: pd.Series, uk: pd.Series) -> str:
    stats = _pair_stats(chicago, uk)
    return (
        f"corr={stats['corr']:.3f}\n"
        f"mean diff={stats['mean']:.3f} N\n"
        f"std diff={stats['std']:.3f} N\n"
        f"mae={stats['mae']:.3f} N"
    )


def load_partial_b(action_json: Path, summary_csv: Path) -> tuple[pd.DataFrame, np.ndarray]:
    action_df = load_action_json(action_json)
    summary_df = load_summary_csv(summary_csv)
    merged = summary_df.merge(action_df[["wire_number", "json_B"]], on="wire_number", how="left")
    merged = merged.dropna(subset=["summary_B"]).copy()
    merged["wire_number"] = merged["wire_number"].astype(int)
    json_values = action_df.sort_values("wire_number")["json_B"].to_numpy(dtype=float)
    return merged, json_values


def evaluate_shift(
    merged: pd.DataFrame, json_values: np.ndarray, shift: int
) -> tuple[pd.DataFrame, dict[str, float | int]]:
    mapped = merged["wire_number"].to_numpy(dtype=int) + shift
    valid = (mapped >= 1) & (mapped <= len(json_values))
    model_df = merged.loc[valid, ["wire_number", "summary_B"]].copy()
    model_df["mapped_wire"] = mapped[valid]
    model_df["json_B"] = json_values[mapped[valid] - 1]
    model_df["residual_B"] = model_df["summary_B"] - model_df["json_B"]
    stats = _pair_stats(model_df["summary_B"], model_df["json_B"])
    stats.update({"shift": shift, "count": int(len(model_df))})
    return model_df, stats


def find_best_shift(
    merged: pd.DataFrame, json_values: np.ndarray, shift_min: int, shift_max: int
) -> tuple[pd.DataFrame, pd.DataFrame, int]:
    rows = []
    for shift in range(shift_min, shift_max + 1):
        _, stats = evaluate_shift(merged, json_values, shift)
        rows.append(stats)
    stats_df = pd.DataFrame(rows)
    best = stats_df.sort_values(["corr", "mae"], ascending=[False, True]).iloc[0]
    return stats_df, stats_df[stats_df["shift"] == 0].copy(), int(best["shift"])


def save_raw_plot(
    model_df: pd.DataFrame,
    output_path: Path,
    title: str,
    chicago_name: str,
    uk_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    chicago = model_df["summary_B"]
    uk = model_df["json_B"]
    wire_numbers = model_df["wire_number"]

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    pairs = [("tab:blue", chicago_name, chicago), ("tab:orange", uk_name, uk)]
    for color, name, values in pairs:
        label = _series_label(name, values)
        axes[0].scatter(wire_numbers, values, s=10, alpha=0.25, color=color)
        axes[0].plot(
            wire_numbers,
            _rolling_mean(values.reset_index(drop=True)),
            linewidth=2,
            color=color,
            label=label,
        )
        axes[1].hist(values, bins=30, alpha=0.35, color=color, label=label)
        axes[1].axvline(values.mean(), color=color, linewidth=1.5)

    axes[0].set_title(title)
    axes[0].set_xlabel("CSV Wire Number")
    axes[0].set_ylabel("Tension (N)")
    axes[0].grid(True, linestyle=":", linewidth=0.5, color="gray")
    axes[0].legend(fontsize=8, loc="upper right")
    axes[0].text(
        0.015,
        0.98,
        _pair_stats_text(chicago, uk),
        transform=axes[0].transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.9},
    )

    axes[1].set_title("Distribution")
    axes[1].set_xlabel("Tension (N)")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, linestyle=":", linewidth=0.5, color="gray")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_residual_comparison_plot(
    baseline_df: pd.DataFrame,
    shifted_df: pd.DataFrame,
    baseline_stats: dict[str, float | int],
    shifted_stats: dict[str, float | int],
    label: str,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    models = [
        ("baseline", baseline_df, baseline_stats, "tab:blue"),
        (f"shift {int(shifted_stats['shift']):+d}", shifted_df, shifted_stats, "tab:green"),
    ]

    for name, df, stats, color in models:
        residual = df["residual_B"]
        legend = (
            f"{name} "
            f"(corr={stats['corr']:.3f}, std={stats['std']:.3f}, "
            f"mae={stats['mae']:.3f}, n={int(stats['count'])})"
        )
        axes[0].scatter(df["wire_number"], residual, s=10, alpha=0.25, color=color)
        axes[0].plot(
            df["wire_number"],
            _rolling_mean(residual.reset_index(drop=True)),
            linewidth=2,
            color=color,
            label=legend,
        )
        axes[1].hist(residual, bins=30, alpha=0.35, color=color, label=legend)
        axes[1].axvline(residual.mean(), color=color, linewidth=1.5)

    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title(f"{label} Side B Residuals by Wire: Baseline vs Best Offset")
    axes[0].set_xlabel("CSV Wire Number")
    axes[0].set_ylabel("Residual (Chicago - UK)")
    axes[0].grid(True, linestyle=":", linewidth=0.5, color="gray")
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual (Chicago - UK)")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, linestyle=":", linewidth=0.5, color="gray")
    axes[1].legend(fontsize=8, loc="upper left")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_shift_scan_plot(stats_df: pd.DataFrame, label: str, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    best = stats_df.sort_values(["corr", "mae"], ascending=[False, True]).iloc[0]

    axes[0].plot(stats_df["shift"], stats_df["corr"], color="tab:blue")
    axes[0].axvline(best["shift"], color="tab:red", linestyle="--", linewidth=1)
    axes[0].set_ylabel("Correlation")
    axes[0].grid(True, linestyle=":", linewidth=0.5, color="gray")

    axes[1].plot(stats_df["shift"], stats_df["std"], color="tab:green")
    axes[1].axvline(best["shift"], color="tab:red", linestyle="--", linewidth=1)
    axes[1].set_ylabel("Std Residual")
    axes[1].grid(True, linestyle=":", linewidth=0.5, color="gray")

    axes[2].plot(stats_df["shift"], stats_df["mae"], color="tab:orange")
    axes[2].axvline(best["shift"], color="tab:red", linestyle="--", linewidth=1)
    axes[2].set_ylabel("MAE")
    axes[2].set_xlabel("Index Shift")
    axes[2].grid(True, linestyle=":", linewidth=0.5, color="gray")

    fig.suptitle(
        f"{label} Side B Offset Scan (best corr shift = {int(best['shift']):+d})"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_comparison_csv(
    baseline_df: pd.DataFrame, shifted_df: pd.DataFrame, output_path: Path
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    baseline = baseline_df.rename(
        columns={
            "mapped_wire": "baseline_mapped_wire",
            "json_B": "baseline_json_B",
            "residual_B": "baseline_residual_B",
            "summary_B": "summary_B",
        }
    )
    shifted = shifted_df.rename(
        columns={
            "mapped_wire": "shifted_mapped_wire",
            "json_B": "shifted_json_B",
            "residual_B": "shifted_residual_B",
            "summary_B": "summary_B_shifted",
        }
    )
    output = baseline.merge(shifted, on="wire_number", how="outer")
    output.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze offset-only models for partial B-side data."
    )
    parser.add_argument("action_json", type=Path)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--label", default="APA Layer")
    parser.add_argument("--shift-min", type=int, default=-80)
    parser.add_argument("--shift-max", type=int, default=80)
    parser.add_argument("--raw-baseline-plot", type=Path, required=True)
    parser.add_argument("--raw-shifted-plot", type=Path, required=True)
    parser.add_argument("--residual-plot", type=Path, required=True)
    parser.add_argument("--shift-scan-plot", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    merged, json_values = load_partial_b(args.action_json, args.summary_csv)
    stats_df, baseline_row, best_shift = find_best_shift(
        merged, json_values, args.shift_min, args.shift_max
    )
    baseline_df, baseline_stats = evaluate_shift(merged, json_values, 0)
    shifted_df, shifted_stats = evaluate_shift(merged, json_values, best_shift)

    save_raw_plot(
        baseline_df,
        args.raw_baseline_plot,
        f"{args.label} Side B Raw Tensions (Current Indexing)",
        "Chicago B",
        "UK B",
    )
    save_raw_plot(
        shifted_df,
        args.raw_shifted_plot,
        f"{args.label} Side B Raw Tensions (Shift {best_shift:+d})",
        "Chicago B",
        f"UK B mapped (shift {best_shift:+d})",
    )
    save_residual_comparison_plot(
        baseline_df,
        shifted_df,
        baseline_stats,
        shifted_stats,
        args.label,
        args.residual_plot,
    )
    save_shift_scan_plot(stats_df, args.label, args.shift_scan_plot)
    save_comparison_csv(baseline_df, shifted_df, args.output_csv)

    summary = pd.DataFrame([baseline_stats, shifted_stats])
    print(f"Best shift by correlation: {best_shift:+d}")
    print(summary.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
