#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-./.venv-jlab/bin/python}"
INTERNALS_ROOT="${INTERNALS_ROOT:-data/eval_adoption_internals_table_filtered}"
RESULTS_ROOT="${RESULTS_ROOT:-results}"
MODEL_NAME="${MODEL_NAME:-deepseek-r1-0528-qwen3-8b}"
TARGET_COL="${TARGET_COL:-absolute_accuracy_decay}"
METHODS="${METHODS:-probe,kernel}"
REDUCED_DIM="${REDUCED_DIM:-0}"
SEEDS="${SEEDS:-42,43,44,45,46}"
NUM_FOLDS="${NUM_FOLDS:-4}"
DEV_FRACTION="${DEV_FRACTION:-0.2}"
NUM_WORKERS="${NUM_WORKERS:-}"
BINARY_EVAL_COL="${BINARY_EVAL_COL:-}"
THRESHOLD="${THRESHOLD:-auto}"
KERNEL="${KERNEL:-rbf}"

slugify() {
  echo "$1" | tr '[:upper:]' '[:lower:]' | tr '/:' '__' | tr -cs 'a-z0-9._-' '_'
}

MODEL_SLUG="$(slugify "$MODEL_NAME")"
TARGET_SLUG="$(slugify "$TARGET_COL")"

discover_views() {
  if [[ -f "$INTERNALS_ROOT/metadata.csv" ]]; then
    printf '%s\n' "$INTERNALS_ROOT"
    return
  fi

  local found=0
  local candidate
  for candidate in "$INTERNALS_ROOT"/*; do
    if [[ -d "$candidate" && -f "$candidate/metadata.csv" ]]; then
      printf '%s\n' "$candidate"
      found=1
    fi
  done

  if [[ "$found" -eq 0 ]]; then
    echo "No internals directories found under $INTERNALS_ROOT" >&2
    exit 1
  fi
}

IFS=',' read -r -a METHOD_LIST <<< "$METHODS"

while IFS= read -r INTERNALS_DIR; do
  VIEW_NAME="$(basename "$INTERNALS_DIR")"

  for METHOD in "${METHOD_LIST[@]}"; do
    METHOD="$(echo "$METHOD" | xargs)"
    RESULT_DIR="$RESULTS_ROOT/eval_adoption_${TARGET_SLUG}_${MODEL_SLUG}_${VIEW_NAME}_${METHOD}"
    if [[ "$REDUCED_DIM" -gt 0 ]]; then
      RESULT_DIR="${RESULT_DIR}_pca${REDUCED_DIM}"
    fi
    RESULT_DIR="${RESULT_DIR}_cv_v1"

    ARGS=(
      scripts/probe.py
      --internals-dir "$INTERNALS_DIR"
      --results-dir "$RESULT_DIR"
      --model-name "$MODEL_NAME"
      --target-col "$TARGET_COL"
      --method "$METHOD"
      --seeds "$SEEDS"
      --num-folds "$NUM_FOLDS"
      --dev-fraction "$DEV_FRACTION"
      --reduced-dim "$REDUCED_DIM"
    )

    if [[ -n "$NUM_WORKERS" ]]; then
      ARGS+=(--num-workers "$NUM_WORKERS")
    fi
    if [[ -n "$BINARY_EVAL_COL" ]]; then
      ARGS+=(--binary-eval-col "$BINARY_EVAL_COL" --threshold "$THRESHOLD")
    fi
    if [[ "$METHOD" == "kernel" ]]; then
      ARGS+=(--kernel "$KERNEL")
    fi

    echo "Running $METHOD for $VIEW_NAME:"
    printf '  %q' "$PYTHON_BIN" "${ARGS[@]}"
    printf '\n'

    "$PYTHON_BIN" "${ARGS[@]}"
  done
done < <(discover_views)
