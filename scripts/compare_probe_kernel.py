#!/usr/bin/env python3
"""
Compare eval-adoption probe and kernel results.

This script loads two result groups, aggregates the full-test metrics across
seed/fold runs, writes a merged summary CSV, and renders side-by-side layer
curves for:

- regression: Pearson correlation and test error
- classification: accuracy and macro F1

Rows are perturbation types. Columns are methods: probe vs kernel.
Within each subplot:
- blue line: `NONE`
- red line: `RANDOMIZATION`
- shaded band: +/- one standard deviation across runs
"""

from __future__ import annotations

import argparse
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
METHOD_ORDER = ["probe", "kernel"]
DEFAULT_CONTROLS = ["NONE", "RANDOMIZATION"]
METRICS = [
    ("full test pearson", "Pearson correlation", "pearson", False),
    ("full test error", "Test error", "error", True),
    ("full test acc", "Accuracy", "acc", False),
    ("full test f1", "Macro F1", "f1", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probe-results-dir", required=True, help="Probe result group directory.")
    parser.add_argument("--kernel-results-dir", required=True, help="Kernel result group directory.")
    parser.add_argument(
        "--output-dir",
        default="plots/probe_vs_kernel",
        help="Directory where figures and summary CSVs are written.",
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


def parse_metrics_path(group_dir: Path, metrics_path: Path, method: str) -> dict | None:
    relative = metrics_path.relative_to(group_dir)
    parts = relative.parts
    if len(parts) < 8:
        return None

    match = PROBE_RE.match(parts[0])
    if match is None:
        return None

    return {
        "method": method,
        "target": match.group("target"),
        "perturbation_type": match.group("perturbation"),
        "layer": int(match.group("layer")),
        "control_task": parts[3].upper(),
        "fold": int(parts[4]),
        "seed": int(parts[5]),
    }


def load_one_group(group_dir: Path, method: str, controls: set[str], target_prefix: str) -> pd.DataFrame:
    rows: list[dict] = []
    for metrics_path in group_dir.rglob("metrics.csv"):
        parsed = parse_metrics_path(group_dir, metrics_path, method=method)
        if parsed is None:
            continue
        if parsed["control_task"] not in controls:
            continue
        if parsed["target"] != target_prefix:
            continue

        df = pd.read_csv(metrics_path)
        if df.empty:
            continue
        metric_row = df.iloc[-1]
        row = {**parsed}
        for metric_col, _, _, _ in METRICS:
            row[metric_col] = metric_row.get(metric_col, np.nan)
        rows.append(row)

    if not rows:
        raise ValueError(f"No matching metrics found in {group_dir}")
    return pd.DataFrame(rows)


def aggregate_metric(df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    frame = df[df[metric_col].notna()].copy()
    if frame.empty:
        return frame
    grouped = (
        frame.groupby(["method", "perturbation_type", "control_task", "layer"], as_index=False)[metric_col]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    grouped = grouped.rename(columns={"mean": "metric_mean", "std": "metric_std", "count": "metric_count"})
    grouped["metric_std"] = grouped["metric_std"].fillna(0.0)
    return grouped


def plot_metric(agg_df: pd.DataFrame, metric_label: str, output_path: Path, log_scale: bool) -> None:
    perturbations = sorted(agg_df["perturbation_type"].unique())
    methods = [method for method in METHOD_ORDER if method in set(agg_df["method"].unique())]

    fig, axes = plt.subplots(
        len(perturbations),
        len(methods),
        figsize=(5.5 * len(methods), 3.4 * len(perturbations)),
        sharex=True,
        squeeze=False,
    )

    for row_idx, perturbation in enumerate(perturbations):
        for col_idx, method in enumerate(methods):
            ax = axes[row_idx][col_idx]
            subset = agg_df[
                (agg_df["perturbation_type"] == perturbation)
                & (agg_df["method"] == method)
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
                ax.plot(x, y, color=color, linewidth=2, label=CONTROL_LABELS[control])
                ax.fill_between(x, lower, upper, color=color, alpha=0.18)

            ax.set_title(f"{perturbation} | {method}")
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
    controls = {chunk.strip().upper() for chunk in args.controls.split(",") if chunk.strip()}
    output_dir = Path(args.output_dir)

    probe_df = load_one_group(
        Path(args.probe_results_dir),
        method="probe",
        controls=controls,
        target_prefix=args.target_prefix,
    )
    kernel_df = load_one_group(
        Path(args.kernel_results_dir),
        method="kernel",
        controls=controls,
        target_prefix=args.target_prefix,
    )
    metrics_df = pd.concat([probe_df, kernel_df], ignore_index=True)
    metrics_df.to_csv(output_dir / "raw_runs.csv", index=False)

    summary_frames: list[pd.DataFrame] = []
    for metric_col, metric_label, slug, log_scale in METRICS:
        agg_df = aggregate_metric(metrics_df, metric_col)
        if agg_df.empty:
            continue
        plot_metric(
            agg_df,
            metric_label=metric_label,
            output_path=output_dir / f"{slug}_probe_vs_kernel.png",
            log_scale=log_scale,
        )
        metric_summary = agg_df.copy()
        metric_summary["metric"] = metric_col
        metric_summary.to_csv(output_dir / f"{slug}_probe_vs_kernel_summary.csv", index=False)
        summary_frames.append(metric_summary)

    if summary_frames:
        pd.concat(summary_frames, ignore_index=True).to_csv(output_dir / "combined_summary.csv", index=False)

    print("Saved comparison outputs to", output_dir)


if __name__ == "__main__":
    main()
