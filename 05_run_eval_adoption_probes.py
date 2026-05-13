"""
Run regression probes on eval-adoption internals.

For each `permutation_type`, this script:
1. creates an outer 80:20 train/test split,
2. carves a validation split out of the training pool for early stopping,
3. trains one linear regression probe per transformer layer and control setup,
4. writes results via the Holmes CSV logger.

The regression target is `absolute_accuracy_decay`.

`permutation_type` is an eval-adoption perturbation label, not a Holmes control
task. We therefore keep it in the probe name and dataset row identifiers, while
running Holmes control-task variants (`NONE`, `RANDOMIZATION`, `PERMUTATION`)
explicitly as a separate sweep dimension.

Dimensionality reduction is optional. When enabled, each split is projected into
a lower-dimensional PCA space fit on the training vectors only. This reduces
hidden states while preserving as much geometry/variance as possible.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from sklearn.decomposition import PCA

HOLMES_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holmes-evaluation", "core")
if HOLMES_CORE not in sys.path:
    sys.path.insert(0, HOLMES_CORE)

from probing_worker import GeneralProbeWorker  # noqa: E402
from utilities.data_loading import ProbingDataset  # noqa: E402

SEED = 42
CONTROL_TASKS = ["NONE", "RANDOMIZATION", "PERMUTATION"]
DEFAULT_REDUCED_DIM = 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--internals-dir",
        default="data/eval_adoption_internals",
        help="Directory containing metadata.csv and layer_XXX.npy files",
    )
    parser.add_argument(
        "--results-dir",
        default="results/eval_adoption_absolute_accuracy_decay",
        help="Directory where probe outputs are written",
    )
    parser.add_argument(
        "--model-name",
        default="eval-adoption-probe",
        help="Name recorded in Holmes run metadata",
    )
    parser.add_argument(
        "--reduced-dim",
        type=int,
        default=DEFAULT_REDUCED_DIM,
        help="Project hidden states to this many PCA dimensions before probing. Use 0 to disable.",
    )
    return parser.parse_args()


def to_inputs(row_ids: list[int], permutation_type: str) -> list[list[tuple[str, int, int, int]]]:
    return [[(f"{permutation_type}__row_{row_id}", 0, 0, len(f"{permutation_type}__row_{row_id}"))] for row_id in row_ids]


def make_split(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    row_ids: list[int],
    permutation_type: str,
    control_task: str,
    reduced_dim: int,
) -> tuple[ProbingDataset, ProbingDataset, ProbingDataset]:
    indices = np.arange(len(labels))

    train_idx, test_idx = train_test_split(indices, test_size=0.20, random_state=SEED)

    if len(train_idx) < 5:
        raise ValueError(
            f"Not enough rows in training pool for permutation_type={permutation_type!r}: {len(train_idx)}"
        )

    dev_fraction_of_train = max(1, round(len(train_idx) * 0.20)) / len(train_idx)
    train_idx, dev_idx = train_test_split(
        train_idx,
        test_size=dev_fraction_of_train,
        random_state=SEED,
    )

    rng = random.Random(SEED)

    def maybe_reduce(
        train_vecs: np.ndarray,
        dev_vecs: np.ndarray,
        test_vecs: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        if reduced_dim <= 0:
            return train_vecs, dev_vecs, test_vecs

        n_components = min(reduced_dim, train_vecs.shape[0], train_vecs.shape[1])
        if n_components < 1:
            raise ValueError(
                f"Cannot fit PCA for permutation_type={permutation_type!r}: "
                f"train shape={train_vecs.shape}"
            )

        pca = PCA(n_components=n_components, svd_solver="full", random_state=SEED)
        train_reduced = pca.fit_transform(train_vecs)
        dev_reduced = pca.transform(dev_vecs)
        test_reduced = pca.transform(test_vecs)
        return train_reduced.astype(np.float32), dev_reduced.astype(np.float32), test_reduced.astype(np.float32)

    def prepare_vectors(idx: np.ndarray) -> tuple[np.ndarray, list[float]]:
        idx = np.asarray(idx)
        vecs = hidden_states[idx]
        lbls = labels[idx].astype(float).tolist()
        if control_task == "RANDOMIZATION":
            rng.shuffle(lbls)
        elif control_task == "PERMUTATION":
            shuffled = list(range(len(idx)))
            rng.shuffle(shuffled)
            vecs = vecs[shuffled]
        return vecs, lbls

    train_vecs, train_lbls = prepare_vectors(train_idx)
    dev_vecs, dev_lbls = prepare_vectors(dev_idx)
    test_vecs, test_lbls = prepare_vectors(test_idx)

    train_vecs, dev_vecs, test_vecs = maybe_reduce(train_vecs, dev_vecs, test_vecs)

    def make_dataset_from_arrays(idx: np.ndarray, vecs: np.ndarray, lbls: list[float]) -> ProbingDataset:
        inputs = to_inputs([row_ids[i] for i in np.asarray(idx).tolist()], permutation_type)
        encoded = list(vecs)
        return ProbingDataset(inputs, encoded, lbls)

    train_ds = make_dataset_from_arrays(train_idx, train_vecs, train_lbls)
    dev_ds = make_dataset_from_arrays(dev_idx, dev_vecs, dev_lbls)
    test_ds = make_dataset_from_arrays(test_idx, test_vecs, test_lbls)

    dev_ds.update_seen(train_ds.unique_inputs)
    test_ds.update_seen(train_ds.unique_inputs)

    return train_ds, dev_ds, test_ds


def run_layer(
    layer_idx: int,
    hidden_states: np.ndarray,
    labels: np.ndarray,
    row_ids: list[int],
    permutation_type: str,
    control_task: str,
    reduced_dim: int,
    n_total_layers: int,
    results_dir: str,
    model_name: str,
) -> None:
    train_ds, dev_ds, test_ds = make_split(
        hidden_states,
        labels,
        row_ids,
        permutation_type,
        control_task,
        reduced_dim,
    )
    hidden_dim = int(np.asarray(train_ds.inputs_encoded[0]).shape[0])
    probe_name = (
        f"absolute_accuracy_decay__{permutation_type}__"
        f"{control_task.lower()}__L{layer_idx:03d}"
    )

    print(
        f"  {permutation_type:>9} | {control_task:>13} | "
        f"layer {layer_idx:03d} | n={len(labels)} | d={hidden_dim}"
    )
    worker = GeneralProbeWorker(
        hyperparameter={
            "seed": SEED,
            "encoding": "full",
            "batch_size": 8,
            "num_labels": 1,
            "num_hidden_layers": 0,
            "input_dim": hidden_dim,
            "output_dim": hidden_dim,
            "hidden_dim": hidden_dim,
            "learning_rate": 1e-3,
            "dropout": 0.1,
            "warmup_rate": 0.1,
            "optimizer": torch.optim.Adam,
            "probe_task_type": "SENTENCE",
            "model_name": model_name,
            "control_task_type": control_task,
            "sample_size": 0,
        },
        train_dataset=train_ds,
        dev_dataset=dev_ds,
        test_dataset=test_ds,
        n_layers=n_total_layers,
        probe_name=probe_name,
        project_prefix="eval-adoption",
        dump_preds=True,
        force=True,
        result_folder=results_dir,
        logging="local",
    )
    worker.run_fold()


def main() -> None:
    args = parse_args()
    internals_dir = Path(args.internals_dir)
    os.makedirs(args.results_dir, exist_ok=True)

    metadata = pd.read_csv(internals_dir / "metadata.csv").sort_values("row_id")
    metadata["absolute_accuracy_decay"] = metadata["absolute_accuracy_decay"].astype(float)

    layer_files = sorted(
        f.name
        for f in internals_dir.iterdir()
        if f.name.startswith("layer_") and f.suffix == ".npy"
    )
    n_layers = len(layer_files)
    print(
        f"Probing {n_layers} layers across "
        f"{metadata['permutation_type'].nunique()} permutation types and "
        f"{len(CONTROL_TASKS)} control settings"
    )

    for permutation_type, subset in metadata.groupby("permutation_type", sort=True):
        subset = subset.reset_index(drop=True)
        row_ids = subset["row_id"].astype(int).tolist()
        labels = subset["absolute_accuracy_decay"].to_numpy(dtype=np.float32)
        subset_indices = subset["row_id"].to_numpy(dtype=int)

        print(f"\nPermutation type: {permutation_type} | rows={len(subset)}")
        for layer_file in layer_files:
            layer_idx = int(layer_file.replace("layer_", "").replace(".npy", ""))
            layer_states = np.load(internals_dir / layer_file)
            subset_states = layer_states[subset_indices]
            for control_task in CONTROL_TASKS:
                run_layer(
                    layer_idx=layer_idx,
                    hidden_states=subset_states,
                    labels=labels,
                    row_ids=row_ids,
                    permutation_type=permutation_type,
                    control_task=control_task,
                    reduced_dim=args.reduced_dim,
                    n_total_layers=n_layers,
                    results_dir=args.results_dir,
                    model_name=args.model_name,
                )

    print(f"\nDone. Results saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
