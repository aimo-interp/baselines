#!/usr/bin/env python3
"""
Build an OOD CSV from a Hugging Face dataset snapshot.

Default use case:
- repo: aimo-interp/aimo-interp-challenge-sample-full
- keep only rows where model_id == qwen3-8b:low

The script downloads the dataset repo snapshot locally, reads parquet files
under `data/`, optionally filters by split name, then writes a CSV (and
optionally parquet) for downstream OOD extraction/probing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd
from huggingface_hub import snapshot_download


DEFAULT_REPO_ID = "aimo-interp/aimo-interp-challenge-sample-full"
DEFAULT_MODEL_ID = "qwen3-8b:low"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--repo-id",
        default=DEFAULT_REPO_ID,
        help="Hugging Face dataset repo id.",
    )
    parser.add_argument(
        "--repo-revision",
        default="main",
        help="Hugging Face dataset revision.",
    )
    parser.add_argument(
        "--model-id",
        default=DEFAULT_MODEL_ID,
        help="Filter rows to this model_id.",
    )
    parser.add_argument(
        "--split",
        default="",
        help="Optional split/path filter, e.g. validation or full. Empty means use all parquet files under data/.",
    )
    parser.add_argument(
        "--snapshot-dir",
        default="data/hf_snapshots/aimo_interp_challenge_sample_full",
        help="Local directory where the dataset snapshot is stored.",
    )
    parser.add_argument(
        "--output-csv",
        required=True,
        help="Output CSV path.",
    )
    parser.add_argument(
        "--output-parquet",
        default="",
        help="Optional output parquet path.",
    )
    return parser.parse_args()


def find_parquet_files(snapshot_dir: Path, split: str) -> list[Path]:
    parquet_files = sorted((snapshot_dir / "data").rglob("*.parquet"))
    if split:
        split_lower = split.lower()
        parquet_files = [path for path in parquet_files if split_lower in str(path).lower()]
    if not parquet_files:
        raise ValueError(
            f"No parquet files found under {snapshot_dir / 'data'}"
            + (f" matching split filter {split!r}" if split else "")
        )
    return parquet_files


def main() -> None:
    args = parse_args()

    snapshot_dir = Path(args.snapshot_dir)
    snapshot_dir.parent.mkdir(parents=True, exist_ok=True)

    resolved_snapshot = snapshot_download(
        repo_id=args.repo_id,
        repo_type="dataset",
        revision=args.repo_revision,
        local_dir=str(snapshot_dir),
        local_dir_use_symlinks=False,
    )
    snapshot_dir = Path(resolved_snapshot)

    parquet_files = find_parquet_files(snapshot_dir, args.split)
    frames = []
    for parquet_file in parquet_files:
        frame = pd.read_parquet(parquet_file)
        frame["source_parquet"] = str(parquet_file.relative_to(snapshot_dir))
        frames.append(frame)

    df = pd.concat(frames, ignore_index=True)
    if "model_id" not in df.columns:
        raise ValueError("Expected a model_id column in the dataset.")

    filtered = df[df["model_id"] == args.model_id].copy().reset_index(drop=True)
    if filtered.empty:
        raise ValueError(
            f"No rows found for model_id={args.model_id!r}"
            + (f" with split filter {args.split!r}" if args.split else "")
        )

    output_csv = Path(args.output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    filtered.to_csv(output_csv, index=False)

    if args.output_parquet:
        output_parquet = Path(args.output_parquet)
        output_parquet.parent.mkdir(parents=True, exist_ok=True)
        filtered.to_parquet(output_parquet, index=False)

    print(f"snapshot_dir: {snapshot_dir}")
    print(f"parquet_files: {len(parquet_files)}")
    print(f"rows_total: {len(df)}")
    print(f"rows_filtered: {len(filtered)}")
    print(f"output_csv: {output_csv}")
    if args.output_parquet:
        print(f"output_parquet: {args.output_parquet}")
    print(f"columns: {list(filtered.columns)}")


if __name__ == "__main__":
    main()
