from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def load_action_json(path: Path) -> pd.DataFrame:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)

    action_data = payload["data"]
    side_a = action_data["measuredTensions_sideA"]
    side_b = action_data["measuredTensions_sideB"]
    wire_count = max(len(side_a), len(side_b))

    return pd.DataFrame(
        {
            "wire_number": range(1, wire_count + 1),
            "json_A": pd.Series(side_a, dtype="float64"),
            "json_B": pd.Series(side_b, dtype="float64"),
        }
    )


def load_summary_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, encoding="utf-8")
    expected = {"wire_number", "A", "B"}
    missing = expected - set(df.columns)
    if missing:
        missing_str = ", ".join(sorted(missing))
        raise ValueError(f"Summary CSV is missing required columns: {missing_str}")

    return df.rename(columns={"A": "summary_A", "B": "summary_B"})


def build_comparison_frame(action_path: Path, summary_path: Path) -> pd.DataFrame:
    action_df = load_action_json(action_path)
    summary_df = load_summary_csv(summary_path)
    comparison = action_df.merge(summary_df, on="wire_number", how="outer")
    comparison = comparison.sort_values("wire_number").reset_index(drop=True)
    comparison["diff_A"] = comparison["summary_A"] - comparison["json_A"]
    comparison["diff_B"] = comparison["summary_B"] - comparison["json_B"]
    return comparison


def _plot_side_difference(ax: plt.Axes, df: pd.DataFrame, side: str) -> None:
    diff_col = f"diff_{side}"
    valid = df.dropna(subset=[diff_col])

    if valid.empty:
        ax.text(
            0.5,
            0.5,
            f"No overlapping side {side} data\nin summary CSV",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title(f"Side {side} Per-Wire Difference")
        ax.set_xlabel("Wire Number")
        ax.set_ylabel("Summary - JSON")
        ax.grid(True, linestyle=":", linewidth=0.5, color="gray")
        return

    ax.scatter(valid["wire_number"], valid[diff_col], s=12, alpha=0.7)
    ax.plot(valid["wire_number"], valid[diff_col], alpha=0.35, linewidth=1)
    ax.axhline(0.0, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"Side {side} Per-Wire Difference")
    ax.set_xlabel("Wire Number")
    ax.set_ylabel("Summary - JSON")
    ax.grid(True, linestyle=":", linewidth=0.5, color="gray")


def save_per_wire_plot(df: pd.DataFrame, output_path: Path, label: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    _plot_side_difference(axes[0], df, "A")
    _plot_side_difference(axes[1], df, "B")
    fig.suptitle(f"{label}: Summary CSV vs JSON Action Differences by Wire")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def _plot_side_distribution(ax: plt.Axes, df: pd.DataFrame, side: str) -> None:
    diff_col = f"diff_{side}"
    valid = df[diff_col].dropna()

    if valid.empty:
        ax.text(
            0.5,
            0.5,
            f"No overlapping side {side} data\nin summary CSV",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=12,
        )
        ax.set_title(f"Side {side} Difference Distribution")
        ax.set_xlabel("Summary - JSON")
        ax.set_ylabel("Count")
        ax.grid(True, linestyle=":", linewidth=0.5, color="gray")
        return

    ax.hist(valid, bins=30, alpha=0.8, edgecolor="white")
    ax.axvline(0.0, color="black", linestyle="--", linewidth=1)
    ax.axvline(valid.mean(), color="tab:red", linestyle="-", linewidth=1.2)
    ax.set_title(f"Side {side} Difference Distribution")
    ax.set_xlabel("Summary - JSON")
    ax.set_ylabel("Count")
    ax.grid(True, linestyle=":", linewidth=0.5, color="gray")


def save_distribution_plot(df: pd.DataFrame, output_path: Path, label: str) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    _plot_side_distribution(axes[0], df, "A")
    _plot_side_distribution(axes[1], df, "B")
    fig.suptitle(f"{label}: Distribution of Summary CSV - JSON Differences")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def summarise_differences(df: pd.DataFrame) -> str:
    lines: list[str] = []
    for side in ("A", "B"):
        diff_col = f"diff_{side}"
        valid = df[diff_col].dropna()
        if valid.empty:
            lines.append(f"Side {side}: no overlapping data")
            continue

        lines.append(
            "Side {side}: count={count}, mean={mean:.6f}, median={median:.6f}, "
            "mean_abs={mean_abs:.6f}, max_abs={max_abs:.6f}".format(
                side=side,
                count=len(valid),
                mean=valid.mean(),
                median=valid.median(),
                mean_abs=valid.abs().mean(),
                max_abs=valid.abs().max(),
            )
        )
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare JSON action tensions with summary CSV tensions."
    )
    parser.add_argument("action_json", type=Path)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument(
        "--label",
        default="APA Comparison",
        help="Label used in plot titles.",
    )
    parser.add_argument(
        "--output-csv",
        type=Path,
        required=True,
        help="Path for the comparison CSV.",
    )
    parser.add_argument(
        "--per-wire-plot",
        type=Path,
        required=True,
        help="Path for the per-wire difference plot.",
    )
    parser.add_argument(
        "--distribution-plot",
        type=Path,
        required=True,
        help="Path for the difference distribution plot.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    comparison = build_comparison_frame(args.action_json, args.summary_csv)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(args.output_csv, index=False)
    save_per_wire_plot(comparison, args.per_wire_plot, args.label)
    save_distribution_plot(comparison, args.distribution_plot, args.label)
    print(summarise_differences(comparison))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
