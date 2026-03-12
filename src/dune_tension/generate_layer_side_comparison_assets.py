from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

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
    return (
        f"corr={chicago.corr(uk):.3f}\n"
        f"mean diff={diff.mean():.3f} N\n"
        f"std diff={diff.std(ddof=0):.3f} N\n"
        f"mae={diff.abs().mean():.3f} N"
    )


def save_raw_plot(
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
    for color, name, values in [
        ("tab:blue", chicago_name, chicago),
        ("tab:orange", uk_name, uk),
    ]:
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


def save_residual_plot(
    comparison: pd.DataFrame, output_path: Path, label: str, sides: list[str]
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(15, 10))
    colors = {"A": "tab:blue", "B": "tab:green"}

    for side in sides:
        subset = comparison[["wire_number", f"diff_{side}"]].dropna().copy()
        residual = subset[f"diff_{side}"]
        legend = (
            f"{side} "
            f"(mean={residual.mean():.3f}, std={residual.std(ddof=0):.3f}, "
            f"mae={residual.abs().mean():.3f}, n={int(residual.size)})"
        )
        axes[0].scatter(subset["wire_number"], residual, s=10, alpha=0.25, color=colors[side])
        axes[0].plot(
            subset["wire_number"],
            _rolling_mean(residual.reset_index(drop=True)),
            linewidth=2,
            color=colors[side],
            label=legend,
        )
        axes[1].hist(residual, bins=30, alpha=0.35, color=colors[side], label=legend)
        axes[1].axvline(residual.mean(), color=colors[side], linewidth=1.5)

    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    axes[0].set_title(f"{label} Residuals by Wire")
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate baseline side-comparison assets for a layer."
    )
    parser.add_argument("action_json", type=Path)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--label", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison = build_comparison_frame(args.action_json, args.summary_csv)
    available_sides = []
    for side in ["A", "B"]:
        subset = comparison[["wire_number", f"summary_{side}", f"json_{side}"]].dropna()
        if subset.empty:
            continue
        available_sides.append(side)
        save_raw_plot(
            subset["wire_number"],
            subset[f"summary_{side}"],
            subset[f"json_{side}"],
            args.output_dir / f"tension_raw_{side}_{args.label.replace(' ', '_')}.png",
            f"{args.label} Side {side} Raw Tensions",
            f"Chicago {side}",
            f"UK {side}",
        )

    if available_sides:
        save_residual_plot(
            comparison,
            args.output_dir / f"tension_residual_{args.label.replace(' ', '_')}.png",
            args.label,
            available_sides,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
