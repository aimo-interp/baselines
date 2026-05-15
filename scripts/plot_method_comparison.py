#!/usr/bin/env python3
"""
Plot direct method comparisons for eval-adoption runs.

The script reads multiple eval-adoption result groups, detects the method from
the result directory name, aggregates layer-wise metrics across seed/fold runs,
and renders one figure per metric.

Layout:
- rows: perturbation types
- columns: methods (`probe`, `kernel`, `cka`, `rsa`) when present

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
DEFAULT_CONTROLS = ["NONE", "RANDOMIZATION"]
METHOD_ORDER = ["probe", "kernel", "cka", "rsa"]
ORIGIN_ORDER = ["input_last_token"]
METRICS = [
    ("full test pearson", "Pearson correlation", "pearson", False),
    ("full test error", "Test error", "error", True),
    ("full test cka", "Centered Kernel Alignment", "cka", False),
    ("full test rsa", "RSA (Spearman)", "rsa", False),
    ("full test acc", "Accuracy", "acc", False),
    ("full test f1", "Macro F1", "f1", False),
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--results-dirs",
        nargs="+",
        required=True,
        help="Explicit result-group directories to compare.",
    )
    parser.add_argument(
        "--output-dir",
        default="plots/method_comparison",
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
    parser.add_argument(
        "--origin",
        default="",
        help="Optional hidden-state origin filter. Default and supported origin is input_last_token.",
    )
    return parser.parse_args()


def classify_origin(group_name: str) -> str:
    lower = group_name.lower()
    for origin in ORIGIN_ORDER:
        if origin in lower:
            return origin
    return "input_last_token"


def classify_method(group_name: str) -> str:
    lower = group_name.lower()
    if "_kernel_" in lower or lower.endswith("_kernel") or "kernel_cv" in lower:
        return "kernel"
    if "_cka_" in lower or lower.endswith("_cka") or "cka_cv" in lower:
        return "cka"
    if "_rsa_" in lower or lower.endswith("_rsa") or "rsa_cv" in lower:
        return "rsa"
    return "probe"


def parse_metrics_path(group_dir: Path, metrics_path: Path) -> dict | None:
    relative = metrics_path.relative_to(group_dir)
    parts = relative.parts
    if len(parts) < 8:
        return None

    probe_name = parts[0]
    match = PROBE_RE.match(probe_name)
    if match is None:
        return None

    return {
        "group_name": group_dir.name,
        "method": classify_method(group_dir.name),
        "origin": classify_origin(group_dir.name),
        "target": match.group("target"),
        "perturbation_type": match.group("perturbation"),
        "layer": int(match.group("layer")),
        "control_task": parts[3].upper(),
        "fold": int(parts[4]),
        "seed": int(parts[5]),
    }


def load_metrics(group_dirs: list[Path], controls: set[str], target_prefix: str, origin_filter: str) -> pd.DataFrame:
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
            if origin_filter and parsed["origin"] != origin_filter:
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
        raise ValueError("No matching metrics were found.")
    return pd.DataFrame(rows)


def aggregate_metrics(metrics_df: pd.DataFrame, metric_col: str) -> pd.DataFrame:
    frame = metrics_df[metrics_df[metric_col].notna()].copy()
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

    n_rows = len(perturbations)
    n_cols = len(methods)
    fig, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(5.5 * n_cols, 3.4 * n_rows),
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
                label = CONTROL_LABELS[control]
                ax.plot(x, y, color=color, linewidth=2, label=label)
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
    group_dirs = [Path(p) for p in args.results_dirs]
    output_dir = Path(args.output_dir)
    controls = {chunk.strip().upper() for chunk in args.controls.split(",") if chunk.strip()}

    metrics_df = load_metrics(
        group_dirs,
        controls=controls,
        target_prefix=args.target_prefix,
        origin_filter=args.origin.strip(),
    )

    for metric_col, metric_label, slug, log_scale in METRICS:
        if metric_col not in metrics_df.columns or not metrics_df[metric_col].notna().any():
            continue
        agg_df = aggregate_metrics(metrics_df, metric_col)
        plot_metric(
            agg_df,
            metric_label=metric_label,
            output_path=output_dir / f"{slug}_method_comparison.png",
            log_scale=log_scale,
        )
        agg_df.to_csv(output_dir / f"{slug}_method_comparison_summary.csv", index=False)

    print("Used result groups:")
    for group_dir in group_dirs:
        print(f"  - {group_dir}")
    print(f"Saved plots to {output_dir}")


if __name__ == "__main__":
    main()
