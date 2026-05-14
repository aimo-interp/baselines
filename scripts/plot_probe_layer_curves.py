#!/usr/bin/env python3
"""
Plot layer-wise probe performance for eval-adoption regression runs.

The script aggregates `full test pearson` and `full test error` across all
seed/fold runs per layer, then renders:

- one figure for Pearson correlation
- one figure for test error

Each figure is arranged as:
- rows: perturbation types
- columns: reduction settings (`full`, `pca10`, `pca50`) when present

Within each subplot:
- blue line: `NONE`
- red line: `RANDOMIZATION`
- shaded band: +/- one standard deviation across runs
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROBE_RE = re.compile(
    r"^(?:eval-adoption-)?(?P<target>.+?)__(?P<perturbation>[^_][^_]*)__(?P<control_in_probe>none|randomization|permutation)__L(?P<layer>\d{3})$",
    re.IGNORECASE,
)
CONTROL_COLORS = {
    "NONE": "#1f77b4",
    "RANDOMIZATION": "#d62728",
}
CONTROL_LABELS = {
    "NONE": "Normal",
    "RANDOMIZATION": "Control",
}
DEFAULT_CONTROLS = ["NONE", "RANDOMIZATION"]
REDUCTION_ORDER = ["full", "pca10", "pca50"]
METRICS = [
    ("full test pearson", "Pearson correlation", "pearson"),
    ("full test error", "Test error", "error"),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-root",
        default="results",
        help="Root directory containing eval-adoption result groups.",
    )
    parser.add_argument(
        "--results-dirs",
        nargs="*",
        default=None,
        help="Optional explicit result-group directories. If omitted, auto-detect latest full/pca10/pca50 groups.",
    )
    parser.add_argument(
        "--output-dir",
        default="results/plots",
        help="Directory where figures are written.",
    )
    parser.add_argument(
        "--controls",
        default="NONE,RANDOMIZATION",
        help="Comma-separated controls to include.",
    )
    parser.add_argument(
        "--target-prefix",
        default="absolute_accuracy_decay",
        help="Only include probe names whose target prefix matches this string.",
    )
    return parser.parse_args()


def classify_reduction(group_name: str) -> str:
    lower = group_name.lower()
    if "pca10" in lower:
        return "pca10"
    if "pca50" in lower:
        return "pca50"
    return "full"


def autodetect_result_dirs(results_root: Path) -> list[Path]:
    candidates = [p for p in results_root.iterdir() if p.is_dir() and p.name.startswith("eval_adoption_")]
    selected: dict[str, Path] = {}
    for reduction in REDUCTION_ORDER:
        reduction_candidates = [p for p in candidates if classify_reduction(p.name) == reduction]
        if reduction_candidates:
            selected[reduction] = max(reduction_candidates, key=lambda p: p.stat().st_mtime)
    return [selected[r] for r in REDUCTION_ORDER if r in selected]


def parse_metrics_path(group_dir: Path, metrics_path: Path) -> dict | None:
    relative = metrics_path.relative_to(group_dir)
    parts = relative.parts
    if len(parts) < 8:
        return None

    probe_name = parts[0]
    match = PROBE_RE.match(probe_name)
    if match is None:
        return None

    control_task = parts[3].upper()
    fold = int(parts[4])
    seed = int(parts[5])

    return {
        "group_name": group_dir.name,
        "reduction": classify_reduction(group_dir.name),
        "probe_name": probe_name,
        "target": match.group("target"),
        "perturbation_type": match.group("perturbation"),
        "layer": int(match.group("layer")),
        "control_task": control_task,
        "fold": fold,
        "seed": seed,
        "metrics_path": metrics_path,
    }


def load_metrics(group_dirs: list[Path], controls: set[str], target_prefix: str) -> pd.DataFrame:
    rows: list[dict] = []
    for group_dir in group_dirs:
        for metrics_path in group_dir.rglob("metrics.csv"):
            parsed = parse_metrics_path(group_dir, metrics_path)
            if parsed is None:
                continue
            if parsed["control_task"] not in controls:
                continue
            if parsed["target"] != target_prefix:
                continue

            df = pd.read_csv(metrics_path)
            if df.empty:
                continue
            row = df[df["full test pearson"].notna() | df["full test error"].notna()]
            metric_row = row.iloc[-1] if not row.empty else df.iloc[-1]

            rows.append(
                {
                    **parsed,
                    "full test pearson": metric_row.get("full test pearson", np.nan),
                    "full test error": metric_row.get("full test error", np.nan),
                }
            )

    if not rows:
        raise ValueError("No matching metrics were found.")

    return pd.DataFrame(rows)


def aggregate_metrics(metrics_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    frame = metrics_df[metrics_df[metric_col].notna()].copy()
    grouped = (
        frame.groupby(["reduction", "perturbation_type", "control_task", "layer"], as_index=False)[metric_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped = grouped.rename(columns={"mean": "metric_mean", "std": "metric_std", "count": "metric_count"})
    grouped["metric_std"] = grouped["metric_std"].fillna(0.0)
    return grouped


def plot_metric(agg_df: pd.DataFrame, metric_label: str, output_path: Path) -> None:
    perturbations = sorted(agg_df["perturbation_type"].unique())
    reductions = [r for r in REDUCTION_ORDER if r in set(agg_df["reduction"].unique())]

    n_rows = len(perturbations)
    n_cols = len(reductions)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 3.4 * n_rows),
        sharex=True,
        squeeze=False,
    )

    for row_idx, perturbation in enumerate(perturbations):
        for col_idx, reduction in enumerate(reductions):
            ax = axes[row_idx][col_idx]
            subset = agg_df[
                (agg_df["perturbation_type"] == perturbation)
                & (agg_df["reduction"] == reduction)
            ]

            for control in DEFAULT_CONTROLS:
                control_df = subset[subset["control_task"] == control].sort_values("layer")
                if control_df.empty:
                    continue
                x = control_df["layer"].to_numpy()
                y = control_df["metric_mean"].to_numpy()
                y_std = control_df["metric_std"].to_numpy()
                color = CONTROL_COLORS[control]
                label = CONTROL_LABELS[control]
                ax.plot(x, y, color=color, linewidth=2, label=label)
                ax.fill_between(x, y - y_std, y + y_std, color=color, alpha=0.18)

            ax.set_title(f"{perturbation} | {reduction}")
            ax.set_xlabel("Layer")
            ax.set_ylabel(metric_label)
            ax.grid(alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    results_root = Path(args.results_root)
    output_dir = Path(args.output_dir)
    controls = {chunk.strip().upper() for chunk in args.controls.split(",") if chunk.strip()}

    if args.results_dirs:
        group_dirs = [Path(p) for p in args.results_dirs]
    else:
        group_dirs = autodetect_result_dirs(results_root)

    if not group_dirs:
        raise ValueError("No result-group directories found.")

    metrics_df = load_metrics(group_dirs, controls=controls, target_prefix=args.target_prefix)

    for metric_col, metric_label, slug in METRICS:
        agg_df = aggregate_metrics(metrics_df, metric_col)
        plot_metric(
            agg_df,
            metric_label=metric_label,
            output_path=output_dir / f"{slug}_layer_curves.png",
        )
        agg_df.to_csv(output_dir / f"{slug}_layer_curves_summary.csv", index=False)

    print("Used result groups:")
    for group_dir in group_dirs:
        print(f"  - {group_dir}")
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
