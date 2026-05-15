#!/usr/bin/env python3
"""
Summarize the best classification accuracy per permutation type.

This script scans one result group, aggregates classification metrics across
seed/fold runs, and selects the best layer for each permutation type. By
default it uses balanced accuracy.
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import numpy as np
import pandas as pd

PROBE_RE = re.compile(
    r"^(?:eval-adoption-)?(?P<target>.+?)__(?P<perturbation>[^_][^_]*)__(?P<control_in_probe>none|randomization)__L(?P<layer>\d{3})$",
    re.IGNORECASE,
)
DEFAULT_CONTROLS = ["NONE", "RANDOMIZATION"]
METRIC_CHOICES = ["full test balanced_acc", "full test acc", "full test f1"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--results-dir", required=True, help="Result group directory to summarize.")
    parser.add_argument(
        "--output-dir",
        default="plots/best_accuracy_summary",
        help="Directory where CSV summaries are written.",
    )
    parser.add_argument(
        "--target-prefix",
        default="is_robust",
        help="Only include probe names whose target prefix matches this string.",
    )
    parser.add_argument(
        "--metric",
        default="full test balanced_acc",
        choices=METRIC_CHOICES,
        help="Classification metric used to pick the best layer.",
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


def load_runs(results_dir: Path, target_prefix: str, controls: set[str], metric: str) -> pd.DataFrame:
    rows: list[dict] = []
    for metrics_path in results_dir.rglob("metrics.csv"):
        parsed = parse_metrics_path(results_dir, metrics_path)
        if parsed is None:
            continue
        if parsed["target"] != target_prefix:
            continue
        if parsed["control_task"] not in controls:
            continue

        df = pd.read_csv(metrics_path)
        if df.empty or metric not in df.columns:
            continue

        metric_row = df.iloc[-1]
        metric_value = metric_row.get(metric, np.nan)
        if pd.isna(metric_value):
            continue

        rows.append(
            {
                **parsed,
                metric: float(metric_value),
            }
        )

    if not rows:
        raise ValueError(f"No matching metrics with column {metric!r} found in {results_dir}")
    return pd.DataFrame(rows)


def aggregate_runs(runs_df: pd.DataFrame, metric: str) -> pd.DataFrame:
    agg_df = (
        runs_df.groupby(["perturbation_type", "control_task", "layer"], as_index=False)[metric]
        .agg(["mean", "std", "count"])
        .reset_index()
    )
    agg_df = agg_df.rename(columns={"mean": "metric_mean", "std": "metric_std", "count": "metric_count"})
    agg_df["metric_std"] = agg_df["metric_std"].fillna(0.0)
    return agg_df


def select_best_layers(agg_df: pd.DataFrame) -> pd.DataFrame:
    best_df = (
        agg_df.sort_values(
            ["perturbation_type", "control_task", "metric_mean", "metric_std", "layer"],
            ascending=[True, True, False, True, True],
        )
        .groupby(["perturbation_type", "control_task"], as_index=False)
        .first()
    )
    return best_df


def main() -> None:
    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    controls = {chunk.strip().upper() for chunk in args.controls.split(",") if chunk.strip()}
    runs_df = load_runs(
        results_dir=results_dir,
        target_prefix=args.target_prefix,
        controls=controls,
        metric=args.metric,
    )
    agg_df = aggregate_runs(runs_df, args.metric)
    best_df = select_best_layers(agg_df)

    runs_df.to_csv(output_dir / "raw_runs.csv", index=False)
    agg_df.to_csv(output_dir / "layer_averages.csv", index=False)
    best_df.to_csv(output_dir / "best_by_permutation_type.csv", index=False)

    print("\nBest layer per permutation type:")
    print(best_df.to_string(index=False))
    print(f"\nWrote: {output_dir / 'best_by_permutation_type.csv'}")


if __name__ == "__main__":
    main()
