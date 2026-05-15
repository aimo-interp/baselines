#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./.venv-jlab/bin/python}"
DATASET_CSV="${DATASET_CSV:-data/math-robust-agg.csv}"
MODEL_ID="${MODEL_ID:-deepseek-ai/DeepSeek-R1-0528-Qwen3-8B}"
OUTPUT_DIR="${OUTPUT_DIR:-data/eval_adoption_internals_table_filtered}"
DEVICE="${DEVICE:-auto}"

ARGS=(
  scripts/encode.py
  --dataset-csv "$DATASET_CSV"
  --model-id "$MODEL_ID"
  --output-dir "$OUTPUT_DIR"
  --device "$DEVICE"
)

echo "Running encoding with:"
printf '  %q' "$PYTHON_BIN" "${ARGS[@]}"
printf '\n'

"$PYTHON_BIN" "${ARGS[@]}"
