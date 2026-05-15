#!/usr/bin/env python3
"""
Plot layer-wise probe performance for eval-adoption probe runs.

The script aggregates task-appropriate full-test metrics across all seed/fold
runs per layer, then renders:

- regression: Pearson correlation and test error
- classification: accuracy and macro F1

Each figure is arranged as:
- rows: perturbation types
- columns: either reduction settings (`full`, `pca10`, `pca50`) or hidden-state
  origins (currently `input_last_token`)

Within each subplot:
- blue line: `NONE`
- red line: `RANDOMIZATION`
- shaded band: +/- one standard deviation across runs
"""

from __future__ import annotations

import argparse
import json
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
ORIGIN_ORDER = ["input_last_token"]
def metric_specs(scope: str) -> tuple[list[tuple[str, str, str]], list[tuple[str, str, str]]]:
    if scope == "ood":
        return (
            [
                ("ood test pearson", "OOD Pearson correlation", "ood_pearson"),
                ("ood test error", "OOD test error", "ood_error"),
            ],
            [
                ("ood test acc", "OOD accuracy", "ood_acc"),
                ("ood test f1", "OOD macro F1", "ood_f1"),
            ],
        )
    return (
        [
            ("full test pearson", "Pearson correlation", "pearson"),
            ("full test error", "Test error", "error"),
        ],
        [
            ("full test acc", "Accuracy", "acc"),
            ("full test f1", "Macro F1", "f1"),
        ],
    )


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
    parser.add_argument(
        "--metric-set",
        default="auto",
        choices=["auto", "regression", "classification"],
        help="Metric family to plot. 'auto' infers from available full-test metric columns.",
    )
    parser.add_argument(
        "--column-mode",
        default="reduction",
        choices=["reduction", "origin"],
        help="Which run dimension to place in subplot columns.",
    )
    parser.add_argument(
        "--metric-scope",
        default="full",
        choices=["full", "ood"],
        help="Plot in-distribution full-test metrics or OOD metrics from ood_metrics.json.",
    )
    return parser.parse_args()


def classify_reduction(group_name: str) -> str:
    lower = group_name.lower()
    if "pca10" in lower:
        return "pca10"
    if "pca50" in lower:
        return "pca50"
    return "full"


def classify_origin(group_name: str) -> str:
    lower = group_name.lower()
    for origin in ORIGIN_ORDER:
        if origin in lower:
            return origin
    return "input_last_token"


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
        "origin": classify_origin(group_dir.name),
        "probe_name": probe_name,
        "target": match.group("target"),
        "perturbation_type": match.group("perturbation"),
        "layer": int(match.group("layer")),
        "control_task": control_task,
        "fold": fold,
        "seed": seed,
        "metrics_path": metrics_path,
    }


def load_metrics(
    group_dirs: list[Path],
    controls: set[str],
    target_prefix: str,
    metric_scope: str,
) -> pd.DataFrame:
    regression_metrics, classification_metrics = metric_specs(metric_scope)
    metric_columns = [name for name, _, _ in regression_metrics + classification_metrics]
    rows: list[dict] = []
    for group_dir in group_dirs:
        pattern = "ood_metrics.json" if metric_scope == "ood" else "metrics.csv"
        for metrics_path in group_dir.rglob(pattern):
            parsed = parse_metrics_path(group_dir, metrics_path)
            if parsed is None:
                continue
            if parsed["control_task"] not in controls:
                continue
            if parsed["target"] != target_prefix:
                continue

            if metric_scope == "ood":
                metric_row = json.loads(metrics_path.read_text())
            else:
                df = pd.read_csv(metrics_path)
                if df.empty:
                    continue
                regression_mask = pd.Series(False, index=df.index)
                for metric_col, _, _ in regression_metrics:
                    if metric_col in df.columns:
                        regression_mask = regression_mask | df[metric_col].notna()
                classification_mask = pd.Series(False, index=df.index)
                for metric_col, _, _ in classification_metrics:
                    if metric_col in df.columns:
                        classification_mask = classification_mask | df[metric_col].notna()
                row = df[regression_mask | classification_mask]
                metric_row = row.iloc[-1].to_dict() if not row.empty else df.iloc[-1].to_dict()

            rows.append({**parsed, **{col: metric_row.get(col, np.nan) for col in metric_columns}})

    if not rows:
        raise ValueError("No matching metrics were found.")

    return pd.DataFrame(rows)


def select_metrics(metrics_df: pd.DataFrame, metric_set: str, metric_scope: str) -> list[tuple[str, str, str]]:
    regression_metrics, classification_metrics = metric_specs(metric_scope)
    if metric_set == "regression":
        return regression_metrics
    if metric_set == "classification":
        return classification_metrics

    has_regression = any(
        col in metrics_df.columns and metrics_df[col].notna().any()
        for col, _, _ in regression_metrics
    )
    has_classification = any(
        col in metrics_df.columns and metrics_df[col].notna().any()
        for col, _, _ in classification_metrics
    )

    if has_regression:
        return regression_metrics
    if has_classification:
        return classification_metrics
    raise ValueError("No supported full-test metrics were found in the selected result groups.")


def aggregate_metrics(metrics_df: pd.DataFrame, metric_col: str, column_mode: str) -> pd.DataFrame:
    frame = metrics_df[metrics_df[metric_col].notna()].copy()
    column_key = "origin" if column_mode == "origin" else "reduction"
    grouped = (
        frame.groupby([column_key, "perturbation_type", "control_task", "layer"], as_index=False)[metric_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped = grouped.rename(columns={"mean": "metric_mean", "std": "metric_std", "count": "metric_count"})
    grouped["metric_std"] = grouped["metric_std"].fillna(0.0)
    return grouped


def plot_metric(
    agg_df: pd.DataFrame,
    metric_label: str,
    output_path: Path,
    column_mode: str,
    log_scale: bool = False,
) -> None:
    perturbations = sorted(agg_df["perturbation_type"].unique())
    column_key = "origin" if column_mode == "origin" else "reduction"
    ordered_values = ORIGIN_ORDER if column_mode == "origin" else REDUCTION_ORDER
    column_values = [value for value in ordered_values if value in set(agg_df[column_key].unique())]

    n_rows = len(perturbations)
    n_cols = len(column_values)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 3.4 * n_rows),
        sharex=True,
        squeeze=False,
    )

    for row_idx, perturbation in enumerate(perturbations):
        for col_idx, column_value in enumerate(column_values):
            ax = axes[row_idx][col_idx]
            subset = agg_df[
                (agg_df["perturbation_type"] == perturbation)
                & (agg_df[column_key] == column_value)
            ]

            for control in DEFAULT_CONTROLS:
                control_df = subset[subset["control_task"] == control].sort_values("layer")
                if control_df.empty:
                    continue
                x = control_df["layer"].to_numpy()
                y = control_df["metric_mean"].to_numpy()
                y_std = control_df["metric_std"].to_numpy()
                if log_scale:
                    eps = 1e-12
                    y = np.clip(y, eps, None)
                    lower = np.clip(y - y_std, eps, None)
                    upper = np.clip(y + y_std, eps, None)
                else:
                    lower = y - y_std
                    upper = y + y_std
                color = CONTROL_COLORS[control]
                label = CONTROL_LABELS[control]
                ax.plot(x, y, color=color, linewidth=2, label=label)
                ax.fill_between(x, lower, upper, color=color, alpha=0.18)

            ax.set_title(f"{perturbation} | {column_value}")
            ax.set_xlabel("Layer")
            ax.set_ylabel(metric_label)
            if log_scale:
                ax.set_yscale("log")
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

    metrics_df = load_metrics(
        group_dirs,
        controls=controls,
        target_prefix=args.target_prefix,
        metric_scope=args.metric_scope,
    )
    selected_metrics = select_metrics(metrics_df, args.metric_set, args.metric_scope)

    for metric_col, metric_label, slug in selected_metrics:
        agg_df = aggregate_metrics(metrics_df, metric_col, column_mode=args.column_mode)
        plot_metric(
            agg_df,
            metric_label=metric_label,
            output_path=output_dir / f"{slug}_layer_curves.png",
            column_mode=args.column_mode,
            log_scale=("error" in slug),
        )
        agg_df.to_csv(output_dir / f"{slug}_layer_curves_summary.csv", index=False)

    print("Used result groups:")
    for group_dir in group_dirs:
        print(f"  - {group_dir}")
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
