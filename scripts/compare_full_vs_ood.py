#!/usr/bin/env python3
"""
Compare in-distribution full-test metrics against OOD metrics for one control.

This script loads a single eval-adoption result group, reads both `metrics.csv`
and `ood_metrics.json`, aggregates them across seed/fold runs, and plots:

- rows: perturbation types
- columns: hidden-state origins or reduction settings inferred from result group names
- lines: full vs OOD for one control task (default: NONE)
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
ORIGIN_ORDER = ["input_last_token"]
REDUCTION_ORDER = ["full", "pca10", "pca50"]
SCOPE_STYLES = {
    "full": {"color": "#1f77b4", "label": "Full test"},
    "ood": {"color": "#2ca02c", "label": "OOD test"},
}
REGRESSION_METRICS = [
    ("full test pearson", "ood test pearson", "Pearson correlation", "pearson", False),
    ("full test error", "ood test error", "Test error", "error", True),
]
CLASSIFICATION_METRICS = [
    ("full test acc", "ood test acc", "Accuracy", "acc", False),
    ("full test f1", "ood test f1", "Macro F1", "f1", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Result group directory to analyze.")
    parser.add_argument(
        "--output-dir",
        default="plots/full_vs_ood",
        help="Directory where figures and summary CSVs are written.",
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
        help="Metric family to compare.",
    )
    parser.add_argument(
        "--control",
        default="NONE",
        help="Single control task to compare across full vs OOD.",
    )
    parser.add_argument(
        "--column-mode",
        default="origin",
        choices=["origin", "reduction"],
        help="Which run dimension to place in subplot columns.",
    )
    return parser.parse_args()


def classify_origin(group_name: str) -> str:
    lower = group_name.lower()
    for origin in ORIGIN_ORDER:
        if origin in lower:
            return origin
    return "input_last_token"


def classify_reduction(group_name: str) -> str:
    lower = group_name.lower()
    if "pca10" in lower:
        return "pca10"
    if "pca50" in lower:
        return "pca50"
    return "full"


def parse_metrics_path(group_dir: Path, metrics_path: Path) -> dict | None:
    relative = metrics_path.relative_to(group_dir)
    parts = relative.parts
    if len(parts) < 8:
        return None
    match = PROBE_RE.match(parts[0])
    if match is None:
        return None
    return {
        "origin": classify_origin(group_dir.name),
        "reduction": classify_reduction(group_dir.name),
        "target": match.group("target"),
        "perturbation_type": match.group("perturbation"),
        "layer": int(match.group("layer")),
        "control_task": parts[3].upper(),
        "fold": int(parts[4]),
        "seed": int(parts[5]),
    }


def load_metrics(results_dir: Path, target_prefix: str, control: str) -> pd.DataFrame:
    rows: list[dict] = []
    for metrics_path in results_dir.rglob("metrics.csv"):
        parsed = parse_metrics_path(results_dir, metrics_path)
        if parsed is None or parsed["target"] != target_prefix or parsed["control_task"] != control:
            continue
        df = pd.read_csv(metrics_path)
        if df.empty:
            continue
        metric_row = df.iloc[-1].to_dict()
        ood_path = metrics_path.with_name("ood_metrics.json")
        ood_row = json.loads(ood_path.read_text()) if ood_path.exists() else {}
        row = {**parsed}
        for full_col, ood_col, _, _, _ in REGRESSION_METRICS + CLASSIFICATION_METRICS:
            row[full_col] = metric_row.get(full_col, np.nan)
            row[ood_col] = ood_row.get(ood_col, np.nan)
        rows.append(row)
    if not rows:
        raise ValueError("No matching metrics were found.")
    return pd.DataFrame(rows)


def select_metric_pairs(df: pd.DataFrame, metric_set: str) -> list[tuple[str, str, str, str, bool]]:
    if metric_set == "regression":
        return REGRESSION_METRICS
    if metric_set == "classification":
        return CLASSIFICATION_METRICS
    has_regression = any(df[full_col].notna().any() or df[ood_col].notna().any() for full_col, ood_col, _, _, _ in REGRESSION_METRICS if full_col in df.columns)
    if has_regression:
        return REGRESSION_METRICS
    return CLASSIFICATION_METRICS


def aggregate_metric(df: pd.DataFrame, full_col: str, ood_col: str, column_mode: str) -> pd.DataFrame:
    column_key = "origin" if column_mode == "origin" else "reduction"
    rows = []
    for scope, metric_col in [("full", full_col), ("ood", ood_col)]:
        frame = df[df[metric_col].notna()].copy()
        if frame.empty:
            continue
        grouped = (
            frame.groupby([column_key, "perturbation_type", "layer"], as_index=False)[metric_col]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        grouped = grouped.rename(columns={"mean": "metric_mean", "std": "metric_std", "count": "metric_count"})
        grouped["metric_std"] = grouped["metric_std"].fillna(0.0)
        grouped["scope"] = scope
        rows.append(grouped)
    if not rows:
        return pd.DataFrame()
    return pd.concat(rows, ignore_index=True)


def plot_metric(agg_df: pd.DataFrame, metric_label: str, output_path: Path, column_mode: str, log_scale: bool) -> None:
    perturbations = sorted(agg_df["perturbation_type"].unique())
    column_key = "origin" if column_mode == "origin" else "reduction"
    ordered_values = ORIGIN_ORDER if column_mode == "origin" else REDUCTION_ORDER
    column_values = [value for value in ordered_values if value in set(agg_df[column_key].unique())]
    fig, axes = plt.subplots(
        len(perturbations),
        len(column_values),
        figsize=(5.5 * len(column_values), 3.4 * len(perturbations)),
        sharex=True,
        squeeze=False,
    )
    for row_idx, perturbation in enumerate(perturbations):
        for col_idx, column_value in enumerate(column_values):
            ax = axes[row_idx][col_idx]
            subset = agg_df[(agg_df["perturbation_type"] == perturbation) & (agg_df[column_key] == column_value)]
            for scope in ["full", "ood"]:
                scope_df = subset[subset["scope"] == scope].sort_values("layer")
                if scope_df.empty:
                    continue
                x = scope_df["layer"].to_numpy()
                y = scope_df["metric_mean"].to_numpy()
                y_std = scope_df["metric_std"].to_numpy()
                if log_scale:
                    eps = 1e-12
                    y = np.clip(y, eps, None)
                    lower = np.clip(y - y_std, eps, None)
                    upper = np.clip(y + y_std, eps, None)
                else:
                    lower = y - y_std
                    upper = y + y_std
                ax.plot(x, y, color=SCOPE_STYLES[scope]["color"], linewidth=2, label=SCOPE_STYLES[scope]["label"])
                ax.fill_between(x, lower, upper, color=SCOPE_STYLES[scope]["color"], alpha=0.18)
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
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    df = load_metrics(results_dir, args.target_prefix, args.control.upper())
    for full_col, ood_col, label, slug, log_scale in select_metric_pairs(df, args.metric_set):
        agg_df = aggregate_metric(df, full_col, ood_col, args.column_mode)
        if agg_df.empty:
            continue
        plot_metric(
            agg_df,
            metric_label=label,
            output_path=output_dir / f"{slug}_full_vs_ood.png",
            column_mode=args.column_mode,
            log_scale=log_scale,
        )
        agg_df.to_csv(output_dir / f"{slug}_full_vs_ood_summary.csv", index=False)
    print(f"Saved comparison outputs to {output_dir}")


if __name__ == "__main__":
    main()
