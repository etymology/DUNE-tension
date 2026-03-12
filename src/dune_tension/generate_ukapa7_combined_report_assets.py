from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from dune_tension.compare_b_index_models import build_model_comparison
from dune_tension.compare_tension_sources import build_comparison_frame


def _rolling_mean(series: pd.Series, window: int = 15) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def _series_label(name: str, values: pd.Series) -> str:
    return (
        f"{name} "
        f"(mean={values.mean():.3f}, std={values.std(ddof=0):.3f}, n={int(values.size)})"
    )


def _pair_stats_text(chicago: pd.Series, uk: pd.Series) -> str:
    diff = chicago - uk
    corr = chicago.corr(uk)
    return (
        f"corr={corr:.3f}\n"
        f"mean diff={diff.mean():.3f} N\n"
        f"std diff={diff.std(ddof=0):.3f} N\n"
        f"mae={diff.abs().mean():.3f} N"
    )


def _save_two_series_plot(
    wire_numbers: pd.Series,
    chicago: pd.Series,
    uk: pd.Series,
    output_path: Path,
    title: str,
    chicago_name: str,
    uk_name: str,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    pairs = [
        ("tab:blue", chicago_name, chicago),
        ("tab:orange", uk_name, uk),
    ]

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


def _save_residual_plot(
    comparison: pd.DataFrame,
    output_path: Path,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    series_map = {
        "A baseline": comparison[["wire_number", "diff_A"]].dropna(),
        "B baseline": comparison[["wire_number", "diff_B"]].dropna(),
    }
    colors = {"A baseline": "tab:blue", "B baseline": "tab:green"}

    for name, subset in series_map.items():
        residual = subset.iloc[:, 1]
        label = (
            f"{name} "
            f"(mean={residual.mean():.3f}, std={residual.std(ddof=0):.3f}, "
            f"mae={residual.abs().mean():.3f}, n={int(residual.size)})"
        )
        axes[0].scatter(
            subset["wire_number"], residual, s=10, alpha=0.25, color=colors[name]
        )
        axes[0].plot(
            subset["wire_number"],
            _rolling_mean(residual.reset_index(drop=True)),
            linewidth=2,
            color=colors[name],
            label=label,
        )
        axes[1].hist(residual, bins=30, alpha=0.35, color=colors[name], label=label)
        axes[1].axvline(residual.mean(), color=colors[name], linewidth=1.5)

    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title("UKAPA7 Layer G Baseline Residuals by Wire")
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


def generate_assets(action_json: Path, summary_csv: Path, output_dir: Path) -> None:
    comparison = build_comparison_frame(action_json, summary_csv)
    b_models, b_stats = build_model_comparison(action_json, summary_csv, -60, 60)
    corrected_model = next(
        model for model in b_stats["model"].tolist() if model.startswith("reversed_shift_")
    )
    corrected_df = b_models[b_models["model"] == corrected_model].copy()
    best_shift = int(
        b_stats.loc[b_stats["model"] == corrected_model, "best_shift"].iloc[0]
    )

    side_a = comparison[["wire_number", "summary_A", "json_A"]].dropna().copy()
    _save_two_series_plot(
        side_a["wire_number"],
        side_a["summary_A"],
        side_a["json_A"],
        output_dir / "tension_raw_A_UKAPA7_G.png",
        "UKAPA7 Layer G Side A Raw Tensions",
        "Chicago A",
        "UK A",
    )

    side_b = comparison[["wire_number", "summary_B", "json_B"]].dropna().copy()
    _save_two_series_plot(
        side_b["wire_number"],
        side_b["summary_B"],
        side_b["json_B"],
        output_dir / "tension_raw_B_baseline_UKAPA7_G.png",
        "UKAPA7 Layer G Side B Raw Tensions (Current Indexing)",
        "Chicago B",
        "UK B",
    )

    _save_two_series_plot(
        corrected_df["wire_number"],
        corrected_df["summary_B"],
        corrected_df["json_B"],
        output_dir / "tension_raw_B_reversed_shifted_UKAPA7_G.png",
        f"UKAPA7 Layer G Side B Raw Tensions (Reversed + Shift {best_shift:+d})",
        "Chicago B",
        f"UK B mapped ({corrected_model})",
    )

    _save_residual_plot(
        comparison,
        output_dir / "tension_residual_baseline_A_B_UKAPA7_G.png",
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate UKAPA7 combined-report plot assets."
    )
    parser.add_argument("action_json", type=Path)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    generate_assets(args.action_json, args.summary_csv, args.output_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
