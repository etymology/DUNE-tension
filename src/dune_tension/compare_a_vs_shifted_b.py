from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from dune_tension.compare_b_index_models import build_model_comparison
from dune_tension.compare_tension_sources import build_comparison_frame


def _rolling_mean(series: pd.Series, window: int = 15) -> pd.Series:
    return series.rolling(window=window, center=True, min_periods=1).mean()


def build_a_vs_shifted_b(
    action_json: Path,
    summary_csv: Path,
    shift_min: int,
    shift_max: int,
    b_model: str,
) -> tuple[pd.DataFrame, pd.DataFrame, int | None]:
    comparison = build_comparison_frame(action_json, summary_csv)
    side_a = (
        comparison[["wire_number", "json_A", "summary_A", "diff_A"]]
        .dropna()
        .copy()
        .rename(
            columns={
                "json_A": "json_value",
                "summary_A": "summary_value",
                "diff_A": "residual",
            }
        )
    )
    side_a["series"] = "A_baseline"
    side_a["mapped_wire"] = side_a["wire_number"]

    b_models, b_stats = build_model_comparison(
        action_json, summary_csv, shift_min=shift_min, shift_max=shift_max
    )
    shifted_model = next(
        model for model in b_stats["model"].tolist() if model.startswith("reversed_shift_")
    )
    if b_model == "best_reversed_shift":
        selected_model = shifted_model
        best_shift_value = b_stats.loc[
            b_stats["model"] == shifted_model, "best_shift"
        ].iloc[0]
        best_shift = int(best_shift_value)
        series_name = f"B_reversed_shift_{best_shift:+d}"
    elif b_model == "reversed":
        selected_model = "reversed"
        best_shift = None
        series_name = "B_reversed"
    else:
        raise ValueError(f"Unsupported b_model: {b_model}")

    side_b = (
        b_models[b_models["model"] == selected_model][
            ["wire_number", "json_B", "summary_B", "residual_B", "mapped_wire"]
        ]
        .copy()
        .rename(
            columns={
                "json_B": "json_value",
                "summary_B": "summary_value",
                "residual_B": "residual",
            }
        )
    )
    side_b["series"] = series_name

    plot_df = pd.concat([side_a, side_b], ignore_index=True)

    stats = []
    for series_name, subset in plot_df.groupby("series"):
        stats.append(
            {
                "series": series_name,
                "count": int(len(subset)),
                "mean": float(subset["residual"].mean()),
                "median": float(subset["residual"].median()),
                "std": float(subset["residual"].std(ddof=0)),
                "mae": float(subset["residual"].abs().mean()),
                "max_abs": float(subset["residual"].abs().max()),
                "negative_frac": float((subset["residual"] < 0).mean()),
            }
        )
    stats_df = pd.DataFrame(stats)
    return plot_df, stats_df, best_shift


def save_plot(plot_df: pd.DataFrame, stats_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    colors = {
        "A_baseline": "tab:blue",
        next(name for name in plot_df["series"].unique() if name.startswith("B_")): "tab:green",
    }

    fig, axes = plt.subplots(2, 1, figsize=(15, 10))

    for series_name in ["A_baseline"] + [
        name for name in plot_df["series"].unique() if name.startswith("B_")
    ]:
        subset = plot_df[plot_df["series"] == series_name].sort_values("wire_number")
        stats = stats_df.loc[stats_df["series"] == series_name].iloc[0]
        label = (
            f"{series_name} "
            f"(mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
            f"mae={stats['mae']:.3f}, n={int(stats['count'])})"
        )
        axes[0].scatter(
            subset["wire_number"],
            subset["residual"],
            s=10,
            alpha=0.25,
            color=colors[series_name],
        )
        axes[0].plot(
            subset["wire_number"],
            _rolling_mean(subset["residual"]),
            linewidth=2,
            color=colors[series_name],
            label=label,
        )
        axes[1].hist(
            subset["residual"],
            bins=30,
            alpha=0.35,
            color=colors[series_name],
            label=label,
        )
        axes[1].axvline(subset["residual"].mean(), color=colors[series_name], linewidth=1.5)

    axes[0].axhline(0.0, color="black", linestyle="--", linewidth=1)
    b_series_name = next(name for name in plot_df["series"].unique() if name.startswith("B_"))
    b_title = b_series_name.replace("_", " ")
    axes[0].set_title(f"UKAPA7 Layer G Residuals by Wire: A Baseline vs {b_title}")
    axes[0].set_xlabel("CSV Wire Number")
    axes[0].set_ylabel("Residual (Chicago - UK)")
    axes[0].grid(True, linestyle=":", linewidth=0.5, color="gray")
    axes[0].legend(fontsize=8, loc="upper right")

    axes[1].axvline(0.0, color="black", linestyle="--", linewidth=1)
    axes[1].set_title("UKAPA7 Layer G Residual Distribution")
    axes[1].set_xlabel("Residual (Chicago - UK)")
    axes[1].set_ylabel("Count")
    axes[1].grid(True, linestyle=":", linewidth=0.5, color="gray")
    axes[1].legend(fontsize=8, loc="upper left")

    fig.suptitle(f"A-Side Baseline vs {b_title}")
    fig.tight_layout()
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def save_csv(plot_df: pd.DataFrame, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output = (
        plot_df.pivot_table(
            index="wire_number",
            columns="series",
            values=["mapped_wire", "json_value", "summary_value", "residual"],
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
        description="Compare side A baseline residuals with reversed+shifted side B residuals."
    )
    parser.add_argument("action_json", type=Path)
    parser.add_argument("summary_csv", type=Path)
    parser.add_argument("--output-plot", type=Path, required=True)
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--shift-min", type=int, default=-60)
    parser.add_argument("--shift-max", type=int, default=60)
    parser.add_argument(
        "--b-model",
        choices=["reversed", "best_reversed_shift"],
        default="best_reversed_shift",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    plot_df, stats_df, best_shift = build_a_vs_shifted_b(
        args.action_json,
        args.summary_csv,
        shift_min=args.shift_min,
        shift_max=args.shift_max,
        b_model=args.b_model,
    )
    save_plot(plot_df, stats_df, args.output_plot)
    save_csv(plot_df, args.output_csv)
    if best_shift is None:
        print("B model: reversed")
    else:
        print(f"Best B reversed shift: {best_shift:+d}")
    print(stats_df.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
