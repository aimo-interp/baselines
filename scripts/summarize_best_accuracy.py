#!/usr/bin/env python3
"""
Summarize and plot accuracy-like metrics per permutation type.

This script scans one or two result groups, aggregates metrics across seed/fold
runs, selects the best layer for each permutation type, and plots layer-wise
accuracy curves. It supports both:

- classification runs via metrics like `full test balanced_acc`
- regression runs via thresholded binary metrics like
  `full test threshold_balanced_accuracy`
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
    r"^(?:eval-adoption-)?(?P<target>.+?)__(?P<perturbation>[^_][^_]*)__(?P<control_in_probe>none|randomization)__L(?P<layer>\d{3})$",
    re.IGNORECASE,
)
DEFAULT_CONTROLS = ["NONE", "RANDOMIZATION"]
METHOD_ORDER = ["probe", "kernel"]
CONTROL_COLORS = {
    "NONE": "#1f77b4",
    "RANDOMIZATION": "#d62728",
}
CONTROL_LABELS = {
    "NONE": "Normal",
    "RANDOMIZATION": "Control",
}
METRIC_CHOICES = [
    "auto",
    "full test threshold_balanced_accuracy",
    "full test threshold_accuracy",
    "full test balanced_acc",
    "full test acc",
    "full test f1",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", default=None, help="Single result group directory to summarize.")
    parser.add_argument("--probe-results-dir", default=None, help="Probe result group directory.")
    parser.add_argument("--kernel-results-dir", default=None, help="Kernel result group directory.")
    parser.add_argument(
        "--output-dir",
        default="plots/best_accuracy_summary",
        help="Directory where CSV summaries are written.",
    )
    parser.add_argument(
        "--target-prefix",
        default="absolute_accuracy_decay",
        help="Only include probe names whose target prefix matches this string.",
    )
    parser.add_argument(
        "--metric",
        default="auto",
        choices=METRIC_CHOICES,
        help="Metric used to pick the best layer. Use 'auto' to prefer thresholded balanced accuracy, then balanced accuracy, then accuracy, then F1.",
    )
    parser.add_argument(
        "--controls",
        default="NONE,RANDOMIZATION",
        help="Comma-separated controls to include.",
    )
    return parser.parse_args()


def parse_metrics_path(results_dir: Path, metrics_path: Path) -> dict | None:
    relative = metrics_path.relative_to(results_dir)
    parts = relative.parts
    if len(parts) < 8:
        return None

    match = PROBE_RE.match(parts[0])
    if match is None:
        return None

    return {
        "target": match.group("target"),
        "perturbation_type": match.group("perturbation"),
        "layer": int(match.group("layer")),
        "control_task": parts[3].upper(),
        "fold": int(parts[4]),
        "seed": int(parts[5]),
    }


def resolve_metric(columns: list[str], requested_metric: str) -> str | None:
    if requested_metric != "auto":
        return requested_metric if requested_metric in columns else None

    candidates = [
        "full test threshold_balanced_accuracy",
        "full test balanced_acc",
        "full test threshold_accuracy",
        "full test acc",
        "full test f1",
    ]
    for metric in candidates:
        if metric in columns:
            return metric
    return None


def infer_method(results_dir: Path, explicit_method: str | None = None) -> str:
    if explicit_method is not None:
        return explicit_method
    lower = results_dir.name.lower()
    if "kernel" in lower:
        return "kernel"
    return "probe"


def load_runs(
    results_dir: Path,
    target_prefix: str,
    controls: set[str],
    requested_metric: str,
    method: str | None = None,
) -> tuple[pd.DataFrame, str]:
    rows: list[dict] = []
    chosen_metric: str | None = None
    resolved_method = infer_method(results_dir, method)
    for metrics_path in results_dir.rglob("metrics.csv"):
        parsed = parse_metrics_path(results_dir, metrics_path)
        if parsed is None:
            continue
        if parsed["target"] != target_prefix:
            continue
        if parsed["control_task"] not in controls:
            continue

        df = pd.read_csv(metrics_path)
        if df.empty:
            continue

        metric = resolve_metric(df.columns.tolist(), requested_metric)
        if metric is None:
            continue
        if chosen_metric is None:
            chosen_metric = metric
        elif metric != chosen_metric:
            continue

        metric_row = df.iloc[-1]
        metric_value = metric_row.get(metric, np.nan)
        if pd.isna(metric_value):
            continue

        rows.append(
            {
                **parsed,
                "method": resolved_method,
                metric: float(metric_value),
            }
        )

    if not rows:
        raise ValueError(
            f"No matching metrics found in {results_dir} for target {target_prefix!r}. "
            f"Tried metric={requested_metric!r}."
        )
    assert chosen_metric is not None
    return pd.DataFrame(rows), chosen_metric


def aggregate_runs(runs_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    agg_df = (
        runs_df.groupby(["method", "perturbation_type", "control_task", "layer"], as_index=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg_df = agg_df.rename(columns={"mean": "metric_mean", "std": "metric_std", "count": "metric_count"})
    agg_df["metric_std"] = agg_df["metric_std"].fillna(0.0)
    return agg_df


def select_best_layers(agg_df: pd.DataFrame) -> pd.DataFrame:
    best_df = (
        agg_df.sort_values(
            ["method", "perturbation_type", "control_task", "metric_mean", "metric_std", "layer"],
            ascending=[True, True, True, False, True, True],
        )
        .groupby(["method", "perturbation_type", "control_task"], as_index=False)
        .first()
    )
    return best_df


def metric_label(metric: str) -> str:
    labels = {
        "full test threshold_balanced_accuracy": "Thresholded Balanced Accuracy",
        "full test threshold_accuracy": "Thresholded Accuracy",
        "full test balanced_acc": "Balanced Accuracy",
        "full test acc": "Accuracy",
        "full test f1": "Macro F1",
    }
    return labels.get(metric, metric)


def plot_best_layers(best_df: pd.DataFrame, resolved_metric: str, output_dir: Path) -> None:
    methods = [method for method in METHOD_ORDER if method in set(best_df["method"].unique())]
    perturbations = sorted(best_df["perturbation_type"].unique())
    x = np.arange(len(perturbations))
    width = 0.38

    fig, axes = plt.subplots(1, len(methods), figsize=(max(8, 6 * len(methods)), 5.5), squeeze=False)

    for col_idx, method in enumerate(methods):
        ax = axes[0][col_idx]
        method_df = best_df[best_df["method"] == method]
        plotted = False

        for idx, control in enumerate(DEFAULT_CONTROLS):
            control_df = (
                method_df[method_df["control_task"] == control]
                .set_index("perturbation_type")
                .reindex(perturbations)
                .reset_index()
            )
            if control_df["metric_mean"].notna().sum() == 0:
                continue

            offset = (-width / 2) if idx == 0 else (width / 2)
            ax.bar(
                x + offset,
                control_df["metric_mean"].fillna(0.0).to_numpy(),
                width=width,
                yerr=control_df["metric_std"].fillna(0.0).to_numpy(),
                capsize=4,
                color=CONTROL_COLORS[control],
                alpha=0.9,
                label=CONTROL_LABELS[control],
            )
            plotted = True

        ax.set_xticks(x)
        ax.set_xticklabels(perturbations, rotation=30, ha="right")
        ax.set_ylabel(metric_label(resolved_metric))
        ax.set_title(f"{method} | Best {metric_label(resolved_metric)}")
        ax.grid(axis="y", alpha=0.25)
        if plotted:
            ax.legend(frameon=False)

    fig.tight_layout()
    fig.savefig(output_dir / "best_by_permutation_type.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_layer_curves(agg_df: pd.DataFrame, resolved_metric: str, output_dir: Path) -> None:
    methods = [method for method in METHOD_ORDER if method in set(agg_df["method"].unique())]
    perturbations = sorted(agg_df["perturbation_type"].unique())

    fig, axes = plt.subplots(
        len(perturbations),
        len(methods),
        figsize=(5.8 * len(methods), 3.5 * len(perturbations)),
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
                lower = y - y_std
                upper = y + y_std
                color = CONTROL_COLORS[control]
                ax.plot(x, y, color=color, linewidth=2, label=CONTROL_LABELS[control])
                ax.fill_between(x, lower, upper, color=color, alpha=0.18)

            ax.set_title(f"{perturbation} | {method}")
            ax.set_xlabel("Layer")
            ax.set_ylabel(metric_label(resolved_metric))
            ax.grid(alpha=0.25)

    handles, labels = axes[0][0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=len(handles), frameon=False)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(output_dir / "accuracy_across_layers.png", dpi=200, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    controls = {chunk.strip().upper() for chunk in args.controls.split(",") if chunk.strip()}
    run_frames: list[pd.DataFrame] = []
    resolved_metric: str | None = None

    source_dirs: list[tuple[Path, str | None]] = []
    if args.results_dir:
        source_dirs.append((Path(args.results_dir), None))
    if args.probe_results_dir:
        source_dirs.append((Path(args.probe_results_dir), "probe"))
    if args.kernel_results_dir:
        source_dirs.append((Path(args.kernel_results_dir), "kernel"))
    if not source_dirs:
        raise ValueError("Provide --results-dir or at least one of --probe-results-dir / --kernel-results-dir.")

    for results_dir, method in source_dirs:
        runs_df_one, metric_one = load_runs(
            results_dir=results_dir,
            target_prefix=args.target_prefix,
            controls=controls,
            requested_metric=args.metric,
            method=method,
        )
        if resolved_metric is None:
            resolved_metric = metric_one
        elif metric_one != resolved_metric:
            raise ValueError(
                f"Resolved metric mismatch across result groups: {resolved_metric!r} vs {metric_one!r}."
            )
        run_frames.append(runs_df_one)

    runs_df = pd.concat(run_frames, ignore_index=True)
    assert resolved_metric is not None
    agg_df = aggregate_runs(runs_df, resolved_metric)
    best_df = select_best_layers(agg_df)

    runs_df.to_csv(output_dir / "raw_runs.csv", index=False)
    agg_df.to_csv(output_dir / "layer_averages.csv", index=False)
    best_df.to_csv(output_dir / "best_by_permutation_type.csv", index=False)
    plot_best_layers(best_df, resolved_metric, output_dir)
    plot_layer_curves(agg_df, resolved_metric, output_dir)

    print(f"\nUsing metric: {resolved_metric}")
    print("\nBest layer per permutation type:")
    print(best_df.to_string(index=False))
    print(f"\nWrote: {output_dir / 'best_by_permutation_type.csv'}")
    print(f"Wrote: {output_dir / 'best_by_permutation_type.png'}")
    print(f"Wrote: {output_dir / 'accuracy_across_layers.png'}")


if __name__ == "__main__":
    main()
