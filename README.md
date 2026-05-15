# AIMO Baseline

This repo contains a minimal eval-adoption workflow:

- [scripts/encode.py](/Users/tresi/Projects/AIMO-baseline/scripts/encode.py): extract last-input-token hidden states from a dataset CSV
- [scripts/probe.py](/Users/tresi/Projects/AIMO-baseline/scripts/probe.py): run layer-wise `probe` or `kernel` experiments on those hidden states
- [run_encoding.bash](/Users/tresi/Projects/AIMO-baseline/run_encoding.bash): repo-root wrapper for encoding
- [run_probing.bash](/Users/tresi/Projects/AIMO-baseline/run_probing.bash): repo-root wrapper for probing

The old mockup pipeline and the multi-view generation-based extraction flow have been removed.

## Setup

Preferred:

```bash
bash scripts/setup_server.sh
```

Manual setup:

```bash
python3 -m venv .venv-jlab
source .venv-jlab/bin/activate
pip install --upgrade pip setuptools wheel
pip install pandas numpy pyarrow scipy scikit-learn matplotlib datasets huggingface_hub ipykernel jupyterlab pytorch-lightning retry transformers
pip install torch torchvision torchaudio
```

If `holmes-evaluation` is missing:

```bash
git clone --branch probe_only https://github.com/Holmes-Benchmark/holmes-evaluation.git
```

## Expected CSV Columns

Required:

- `problem_id`
- `original_problem`
- `permutation_type`
- `absolute_accuracy_decay`

Preserved in `metadata.csv` when present:

- `model_id`
- `dataset_id`
- `model_is_robust`

## Quickstart

Encode:

```bash
bash run_encoding.bash
```

Probe:

```bash
bash run_probing.bash
```

Useful overrides:

```bash
DATASET_CSV=dataset_as_table_filtered.csv \
OUTPUT_DIR=data/eval_adoption_internals_table_filtered \
bash run_encoding.bash
```

```bash
INTERNALS_ROOT=data/eval_adoption_internals_table_filtered \
MODEL_NAME=deepseek-r1-0528-qwen3-8b \
TARGET_COL=absolute_accuracy_decay \
METHODS=probe,kernel \
bash run_probing.bash
```

## Encoding

The encoder performs a single forward pass per unique prompt and extracts only the final input-token hidden state from every layer.

Direct invocation:

```bash
python scripts/encode.py \
  --dataset-csv dataset_as_table_filtered.csv \
  --model-id deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --output-dir data/eval_adoption_internals_table_filtered \
  --device cuda
```

Output layout:

- `metadata.csv`
- `layer_000.npy`
- `layer_001.npy`
- ...

Notes:

- duplicate prompts are encoded once and reused
- there is no generation step
- only `input_last_token` is supported

## Probing

Supported methods:

- `probe`
- `kernel`

Default sweep settings:

- controls: `NONE`, `RANDOMIZATION`
- seeds: `42,43,44,45,46`
- folds: `4`
- workers: CPU core count

Linear probe:

```bash
python scripts/probe.py \
  --internals-dir data/eval_adoption_internals_table_filtered \
  --results-dir results/eval_adoption_absolute_accuracy_decay_probe_v1 \
  --model-name deepseek-r1-0528-qwen3-8b \
  --target-col absolute_accuracy_decay \
  --method probe
```

Kernel baseline:

```bash
python scripts/probe.py \
  --internals-dir data/eval_adoption_internals_table_filtered \
  --results-dir results/eval_adoption_absolute_accuracy_decay_kernel_v1 \
  --model-name deepseek-r1-0528-qwen3-8b \
  --target-col absolute_accuracy_decay \
  --method kernel \
  --kernel rbf
```

The probe runner skips already completed runs individually when the corresponding `done/metrics.csv` already exists.

## PCA

PCA is optional and applied per split using only the training pool for fitting.

Example:

```bash
python scripts/probe.py \
  --internals-dir data/eval_adoption_internals_table_filtered \
  --results-dir results/eval_adoption_absolute_accuracy_decay_kernel_pca10_v1 \
  --model-name deepseek-r1-0528-qwen3-8b \
  --target-col absolute_accuracy_decay \
  --method kernel \
  --reduced-dim 10
```

Disable PCA with:

```bash
--reduced-dim 0
```

## Plotting

Layer curves:

```bash
MPLCONFIGDIR=/tmp/mpl python scripts/plot_probe_layer_curves.py \
  --results-dirs results/eval_adoption_absolute_accuracy_decay_probe_v1 \
  --target-prefix absolute_accuracy_decay \
  --metric-set regression \
  --column-mode origin \
  --output-dir plots/layer_curves
```

Probe vs kernel:

```bash
MPLCONFIGDIR=/tmp/mpl python scripts/compare_probe_kernel.py \
  --probe-results-dir results/eval_adoption_absolute_accuracy_decay_probe_v1 \
  --kernel-results-dir results/eval_adoption_absolute_accuracy_decay_kernel_v1 \
  --target-prefix absolute_accuracy_decay \
  --output-dir plots/probe_vs_kernel
```

Method comparison:

```bash
MPLCONFIGDIR=/tmp/mpl python scripts/plot_method_comparison.py \
  --results-dirs \
    results/eval_adoption_absolute_accuracy_decay_probe_v1 \
    results/eval_adoption_absolute_accuracy_decay_kernel_v1 \
  --target-prefix absolute_accuracy_decay \
  --origin input_last_token \
  --output-dir plots/method_comparison
```

## Notes

- generated data under `data/`, `results/`, `logs/`, and `plots/` is not meant to be committed by default
- local macOS semaphore limits may make high-worker runs fail; servers are the intended execution target for large sweeps
