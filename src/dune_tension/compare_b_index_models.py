from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dune_tension.compare_tension_sources import load_action_json, load_summary_csv


def _reverse_anchor(wire_numbers: np.ndarray) -> int:
    return int(np.min(wire_numbers) + np.max(wire_numbers))


def _evaluate_mapping(
    model_name: str,
    wire_numbers: np.ndarray,
    summary_values: np.ndarray,
    json_values: np.ndarray,
    mapped_wires: np.ndarray,
) -> tuple[pd.DataFrame, dict[str, float | int | str]]:
    valid = (mapped_wires >= 1) & (mapped_wires <= len(json_values))
    model_df = pd.DataFrame(
        {
            "wire_number": wire_numbers[valid],
            "summary_B": summary_values[valid],
            "mapped_wire": mapped_wires[valid],
            "json_B": json_values[mapped_wires[valid] - 1],
        }
    )
    model_df["model"] = model_name
    model_df["residual_B"] = model_df["summary_B"] - model_df["json_B"]

    stats = {
        "model": model_name,
        "count": int(len(model_df)),
        "corr": float(model_df["summary_B"].corr(model_df["json_B"])),
        "mean": float(model_df["residual_B"].mean()),
        "median": float(model_df["residual_B"].median()),
        "std": float(model_df["residual_B"].std(ddof=0)),
        "mae": float(model_df["residual_B"].abs().mean()),
        "rmse": float(np.sqrt(np.mean(model_df["residual_B"] ** 2))),
        "negative_frac": float((model_df["residual_B"] < 0).mean()),
    }
    return model_df, stats


def _find_best_reversed_shift(
    wire_numbers: np.ndarray,
    summary_values: np.ndarray,
    json_values: np.ndarray,
    shift_min: int,
    shift_max: int,
) -> tuple[int, pd.DataFrame, dict[str, float | int | str]]:
    best_key: tuple[float, float, float, int] | None = None
    best_shift = 0
    best_df: pd.DataFrame | None = None
    best_stats: dict[str, float | int | str] | None = None

    anchor = _reverse_anchor(wire_numbers)
    for shift in range(shift_min, shift_max + 1):
        mapped = anchor - wire_numbers + shift
        model_df, stats = _evaluate_mapping(
            f"reversed_shift_{shift:+d}",
            wire_numbers,
            summary_values,
            json_values,
            mapped,
        )
        key = (
            float(stats["corr"]),
            -float(stats["mae"]),
            -float(stats["std"]),
            int(stats["count"]),
        )
        if best_key is None or key > best_key:
            best_key = key
            best_shift = shift
            best_df = model_df
            best_stats = stats

    assert best_df is not None
    assert best_stats is not None
    return best_shift, best_df, best_stats


def build_model_comparison(
    action_json: Path,
    summary_csv: Path,
    shift_min: int,
    shift_max: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    action_df = load_action_json(action_json)
    summary_df = load_summary_csv(summary_csv)

    merged = summary_df.merge(action_df[["wire_number", "json_B"]], on="wire_number")
    merged = merged.dropna(subset=["summary_B", "json_B"]).copy()
    merged["wire_number"] = merged["wire_number"].astype(int)

    wire_numbers = merged["wire_number"].to_numpy()
    summary_values = merged["summary_B"].to_numpy(dtype=float)
    json_values = action_df.sort_values("wire_number")["json_B"].to_numpy(dtype=float)
    anchor = _reverse_anchor(wire_numbers)

    baseline_df, baseline_stats = _evaluate_mapping(
        "baseline",
        wire_numbers,
        summary_values,
        json_values,
        wire_numbers.copy(),
    )
    reversed_df, reversed_stats = _evaluate_mapping(
        "reversed",
        wire_numbers,
        summary_values,
        json_values,
        anchor - wire_numbers,
    )
    best_shift, shifted_df, shifted_stats = _find_best_reversed_shift(
        wire_numbers,
        summary_values,
        json_values,
        shift_min,
        shift_max,
    )
    shifted_stats["best_shift"] = best_shift

    stats_df = pd.DataFrame([baseline_stats, reversed_stats, shifted_stats])
    model_df = pd.concat([baseline_df, reversed_df, shifted_df], ignore_index=True)
    model_df["best_shift"] = np.where(
        model_df["model"] == shifted_stats["model"], best_shift, np.nan
    )
    return model_df, stats_df


def _rolling_mean(series: pd.Series, window: int = 15) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def save_plot(
    model_df: pd.DataFrame, stats_df: pd.DataFrame, output_path: Path, label: str
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "baseline": "tab:blue",
        "reversed": "tab:orange",
    }
    shifted_name = next(
        model for model in stats_df["model"].tolist() if model.startswith("reversed_shift_")
    )
    colors[shifted_name] = "tab:green"

    fig, axes = plt.subplots(2, 1, figsize=(15, 10), sharex=False)

    for model_name in ["baseline", "reversed", shifted_name]:
        subset = model_df[model_df["model"] == model_name].sort_values("wire_number")
        stats = stats_df.loc[stats_df["model"] == model_name].iloc[0]
        label = (
            f"{model_name} "
            f"(corr={stats['corr']:.3f}, std={stats['std']:.3f}, "
            f"mae={stats['mae']:.3f}, n={int(stats['count'])})"
        )
        axes[0].scatter(
            subset["wire_number"],
            subset["residual_B"],
            s=10,
            alpha=0.25,
            color=colors[model_name],
        )
        axes[0].plot(
            subset["wire_number"],
            _rolling_mean(subset["residual_B"]),
            linewidth=2,
            color=colors[model_name],
            label=label,
        )
        axes[1].hist(
            subset["residual_B"],
            bins=30,
            alpha=0.35,
            color=colors[model_name],
            label=label,
        )
        axes[1].axvline(
            subset["residual_B"].mean(),
            color=colors[model_name],
            linewidth=1.5,
        )

    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title(f"{label} Side B Residuals by Wire")
    axes[0].set_xlabel("CSV Wire Number")
    axes[0].set_ylabel("Residual (Chicago - UK)")
    axes[0].grid(True, linestyle=":", linewidth=0.5, color="gray")
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title(f"{label} Side B Residual Distribution")
    axes[1].set_xlabel("Residual (Chicago - UK)")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, linestyle=":", linewidth=0.5, color="gray")
    axes[1].legend(fontsize=8, loc="upper left")

    fig.suptitle(
        f"{label} B-Side Index Mapping Comparison: Baseline vs Reversed vs Reversed + Best Shift"
    )
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_comparison_csv(model_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = (
        model_df.pivot_table(
            index="wire_number",
            columns="model",
            values=["mapped_wire", "json_B", "residual_B"],
            aggfunc="first",
        )
        .sort_index(axis=1)
        .reset_index()
    )
    output.columns = [
        "wire_number"
        if col == ("wire_number", "")
        else f"{col[1]}_{col[0]}"
        for col in output.columns
    ]
    output.to_csv(output_path, index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare B-side index mapping models for UKAPA7-style data."
    )
    parser.add_argument("action_json", type=Path)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output-plot", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--shift-min", type=int, default=-60)
    parser.add_argument("--shift-max", type=int, default=60)
    parser.add_argument("--label", default="APA Layer")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    model_df, stats_df = build_model_comparison(
        args.action_json, args.summary_csv, args.shift_min, args.shift_max
    )
    save_plot(model_df, stats_df, args.output_plot, args.label)
    save_comparison_csv(model_df, args.output_csv)
    print(stats_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
