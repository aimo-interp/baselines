<h1 align="center">AIMO Interpretability Challenge - Baselines</h1>

<div style="text-align: center; width: 100%;">
  <div style="display: inline-block; text-align: left; width: 100%;">
    <img src="assets/imgs/aimo_interp_challenge.png" style="width: 100%;" alt="Challenge theme">
    <p style="color: gray; font-size: small; margin: 0;">
    </p>
  </div>
</div>

<br>

This repository contains baselines for the AIMO Interpretability Challenge 2026.

*The AIMO Interpretability Challenge is a competition on distinguishing robust from spurious reasoning in frontier mathematical language models. The challenge is motivated by a central limitation of standard reasoning benchmarks: strong final-answer accuracy does not reveal whether a model genuinely relies on stable reasoning mechanisms or exploits brittle shortcuts. Building on the AI Mathematical Olympiad (AIMO) submissions and the Fields Model Initiative’s resources, the competition will provide (1) olympiad-level math problems, (2) their symbolic representations allowing generation of counterfactual variants, (3) access to best-performing AIMO models, and (4) a generous compute of up to 128 H200 GPUs. Based on these, participants will develop methods that identify which model is robust, using models’ internal representations. Our competition will also create a new, open robustness benchmark and baseline systems, aiming to provide a lasting infrastructure for standard benchmarking in interpretability. Scientifically, the competition bridges the gap between the fields of interpretability and generalization by aligning their objectives, while lastingly supporting work aiming to answer the pertaining question in AI research: can we tell if, and to what extent, is the decision making of frontier AI models generalizable, and thus, reliable?*

The current codebase focuses on a minimal representation-based workflow:

- encode the last input-token hidden state from every transformer layer
- predict `absolute_accuracy_decay` from those layer representations
- compare linear probing and kernel baselines

## Table of Contents
- [Table of Contents](#table-of-contents)
- [Setup](#setup)
- [Representation Encoding](#representation-encoding)
- [Layer-wise Prediction](#layer-wise-prediction)
- [Plotting](#plotting)
- [License](#license)
- [Citation](#citation)

## Setup

All code was developed and tested in a Python virtual environment with the dependencies listed in [requirements.txt](/Users/tresi/Projects/AIMO-baseline/requirements.txt).

For a server-style setup, use:

```bash
bash scripts/setup_server.sh
```

Manual setup:

```bash
python3 -m venv .venv-jlab
source .venv-jlab/bin/activate
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

If `holmes-evaluation` is missing:

```bash
git clone --branch probe_only https://github.com/Holmes-Benchmark/holmes-evaluation.git
```

## Representation Encoding

The encoding stage performs a single forward pass per unique prompt and extracts only the final input-token hidden state from every layer. Duplicate prompts are encoded once and then expanded back to the full dataset rows.

Expected CSV columns:

- `problem_id`
- `original_problem`
- `permutation_type`
- `absolute_accuracy_decay`

Direct run:

```bash
python scripts/encode.py \
  --dataset-csv data/math-robust-agg.csv \
  --model-id deepseek-ai/DeepSeek-R1-0528-Qwen3-8B \
  --output-dir data/eval_adoption_internals_table_filtered \
  --device cuda
```

Wrapper:

```bash
bash run_encoding.bash
```

Example override:

```bash
DATASET_CSV=data/math-robust-agg.csv \
OUTPUT_DIR=data/eval_adoption_internals_table_filtered \
bash run_encoding.bash
```

The output directory contains:

- `metadata.csv`
- `layer_000.npy`
- `layer_001.npy`
- ...

## Layer-wise Prediction

The probing stage runs repeated cross-validation over layers and perturbation types using two methods:

- `probe`: linear predictive baseline
- `kernel`: kernel baseline

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

Wrapper:

```bash
bash run_probing.bash
```

Example override:

```bash
INTERNALS_ROOT=data/eval_adoption_internals_table_filtered \
MODEL_NAME=deepseek-r1-0528-qwen3-8b \
TARGET_COL=absolute_accuracy_decay \
METHODS=probe,kernel \
bash run_probing.bash
```

PCA is optional and fit only on the training pool of each split:

```bash
python scripts/probe.py \
  --internals-dir data/eval_adoption_internals_table_filtered \
  --results-dir results/eval_adoption_absolute_accuracy_decay_kernel_pca10_v1 \
  --model-name deepseek-r1-0528-qwen3-8b \
  --target-col absolute_accuracy_decay \
  --method kernel \
  --reduced-dim 10
```

The runner skips already completed tasks individually whenever the corresponding `done/metrics.csv` already exists.

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

## License

No standalone license file is currently included in this repository. Confirm licensing before redistribution or external reuse.

## Citation

No citation metadata is currently included in this repository. If you need a formal citation block, add it together with the corresponding paper, report, or project page.
