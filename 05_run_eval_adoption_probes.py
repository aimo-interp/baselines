"""
Run regression probes on eval-adoption internals.

For each `permutation_type`, this script:
1. runs a repeated K-fold evaluation across configurable seeds and folds,
2. carves a validation split out of each fold's training pool for early stopping,
3. trains one linear regression probe per transformer layer and control setup,
4. optionally evaluates the trained probe on an additional OOD internals set,
5. writes results via the Holmes CSV logger.

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
import json
import os
import random
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold, train_test_split

HOLMES_CORE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "holmes-evaluation", "core")
if HOLMES_CORE not in sys.path:
    sys.path.insert(0, HOLMES_CORE)

from probing_worker import GeneralProbeWorker  # noqa: E402
from utilities.data_loading import ProbingDataset  # noqa: E402

CONTROL_TASKS = ["NONE", "RANDOMIZATION", "PERMUTATION"]
DEFAULT_REDUCED_DIM = 0
DEFAULT_SEEDS = [42, 43, 44, 45, 46]
DEFAULT_NUM_FOLDS = 4
DEFAULT_DEV_FRACTION = 0.20


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
    parser.add_argument(
        "--seeds",
        default=",".join(str(seed) for seed in DEFAULT_SEEDS),
        help="Comma-separated random seeds used for repeated K-fold evaluation.",
    )
    parser.add_argument(
        "--num-folds",
        type=int,
        default=DEFAULT_NUM_FOLDS,
        help="Number of cross-validation folds per seed.",
    )
    parser.add_argument(
        "--dev-fraction",
        type=float,
        default=DEFAULT_DEV_FRACTION,
        help="Fraction of each fold's training pool reserved for validation.",
    )
    parser.add_argument(
        "--ood-internals-dir",
        default="",
        help="Optional directory containing a second metadata.csv plus layer_XXX.npy files used as an additional OOD test set.",
    )
    return parser.parse_args()


def parse_seeds(seed_arg: str) -> list[int]:
    seeds = [int(chunk.strip()) for chunk in seed_arg.split(",") if chunk.strip()]
    if not seeds:
        raise ValueError("At least one seed must be provided")
    return seeds


def to_inputs(
    row_ids: list[int],
    permutation_type: str,
    prefix: str = "id",
) -> list[list[tuple[str, int, int, int]]]:
    return [
        [(f"{prefix}__{permutation_type}__row_{row_id}", 0, 0, len(f"{prefix}__{permutation_type}__row_{row_id}"))]
        for row_id in row_ids
    ]


def load_internals(internals_dir: Path) -> tuple[pd.DataFrame, list[str]]:
    metadata = pd.read_csv(internals_dir / "metadata.csv").sort_values("row_id").reset_index(drop=True)
    metadata["absolute_accuracy_decay"] = metadata["absolute_accuracy_decay"].astype(float)
    layer_files = sorted(
        f.name
        for f in internals_dir.iterdir()
        if f.name.startswith("layer_") and f.suffix == ".npy"
    )
    return metadata, layer_files


def apply_control(
    vecs: np.ndarray,
    lbls: np.ndarray,
    control_task: str,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = random.Random(seed)
    vecs = np.asarray(vecs)
    lbls = np.asarray(lbls, dtype=np.float32)
    if control_task == "RANDOMIZATION":
        shuffled_labels = lbls.tolist()
        rng.shuffle(shuffled_labels)
        return vecs, np.asarray(shuffled_labels, dtype=np.float32)
    if control_task == "PERMUTATION":
        shuffled = list(range(len(vecs)))
        rng.shuffle(shuffled)
        return vecs[shuffled], lbls
    return vecs, lbls


def maybe_reduce(
    train_vecs: np.ndarray,
    dev_vecs: np.ndarray,
    test_vecs: np.ndarray,
    reduced_dim: int,
    seed: int,
    permutation_type: str,
    ood_vecs: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    if reduced_dim <= 0:
        return train_vecs, dev_vecs, test_vecs, ood_vecs

    n_components = min(reduced_dim, train_vecs.shape[0], train_vecs.shape[1])
    if n_components < 1:
        raise ValueError(
            f"Cannot fit PCA for permutation_type={permutation_type!r}: "
            f"train shape={train_vecs.shape}"
        )

    pca = PCA(n_components=n_components, svd_solver="full", random_state=seed)
    train_reduced = pca.fit_transform(train_vecs).astype(np.float32)
    dev_reduced = pca.transform(dev_vecs).astype(np.float32)
    test_reduced = pca.transform(test_vecs).astype(np.float32)
    ood_reduced = pca.transform(ood_vecs).astype(np.float32) if ood_vecs is not None else None
    return train_reduced, dev_reduced, test_reduced, ood_reduced


def build_dataset(
    row_ids: list[int],
    permutation_type: str,
    vecs: np.ndarray,
    lbls: np.ndarray,
    prefix: str,
) -> ProbingDataset:
    inputs = to_inputs(row_ids, permutation_type, prefix=prefix)
    encoded = list(np.asarray(vecs, dtype=np.float32))
    return ProbingDataset(inputs, encoded, np.asarray(lbls, dtype=np.float32).tolist())


def make_fold_datasets(
    hidden_states: np.ndarray,
    labels: np.ndarray,
    row_ids: list[int],
    permutation_type: str,
    control_task: str,
    reduced_dim: int,
    train_pool_idx: np.ndarray,
    test_idx: np.ndarray,
    dev_fraction: float,
    seed: int,
    ood_hidden_states: np.ndarray | None = None,
    ood_labels: np.ndarray | None = None,
    ood_row_ids: list[int] | None = None,
) -> tuple[ProbingDataset, ProbingDataset, ProbingDataset, ProbingDataset | None]:
    train_pool_idx = np.asarray(train_pool_idx)
    test_idx = np.asarray(test_idx)

    if len(train_pool_idx) < 2:
        raise ValueError(
            f"Not enough rows in training pool for permutation_type={permutation_type!r}: {len(train_pool_idx)}"
        )

    dev_count = max(1, round(len(train_pool_idx) * dev_fraction))
    dev_count = min(dev_count, len(train_pool_idx) - 1)
    dev_fraction_of_train = dev_count / len(train_pool_idx)
    train_idx, dev_idx = train_test_split(
        train_pool_idx,
        test_size=dev_fraction_of_train,
        random_state=seed,
    )

    train_vecs, train_lbls = apply_control(hidden_states[train_idx], labels[train_idx], control_task, seed)
    dev_vecs, dev_lbls = apply_control(hidden_states[dev_idx], labels[dev_idx], control_task, seed)
    test_vecs, test_lbls = apply_control(hidden_states[test_idx], labels[test_idx], control_task, seed)

    ood_vecs = ood_lbls = None
    if ood_hidden_states is not None and ood_labels is not None:
        ood_vecs, ood_lbls = apply_control(ood_hidden_states, ood_labels, control_task, seed)

    train_vecs, dev_vecs, test_vecs, ood_vecs = maybe_reduce(
        train_vecs,
        dev_vecs,
        test_vecs,
        reduced_dim,
        seed,
        permutation_type,
        ood_vecs=ood_vecs,
    )

    train_ds = build_dataset([row_ids[i] for i in train_idx.tolist()], permutation_type, train_vecs, train_lbls, prefix="train")
    dev_ds = build_dataset([row_ids[i] for i in dev_idx.tolist()], permutation_type, dev_vecs, dev_lbls, prefix="dev")
    test_ds = build_dataset([row_ids[i] for i in test_idx.tolist()], permutation_type, test_vecs, test_lbls, prefix="test")

    dev_ds.update_seen(train_ds.unique_inputs)
    test_ds.update_seen(train_ds.unique_inputs)

    ood_ds = None
    if ood_vecs is not None and ood_lbls is not None and ood_row_ids is not None:
        ood_ds = build_dataset(ood_row_ids, permutation_type, ood_vecs, ood_lbls, prefix="ood")
        ood_ds.update_seen(train_ds.unique_inputs)

    return train_ds, dev_ds, test_ds, ood_ds


def run_worker(worker: GeneralProbeWorker) -> tuple[str, pd.DataFrame, torch.nn.Module]:
    logger = worker.get_logger()

    if worker.logging == "local":
        log_dir = logger.log_dir
        result_log_dir = log_dir
        if os.path.exists(f"{logger.root_dir}/done") and not worker.force:
            print(f"Already done at {logger.root_dir}/done")
            return f"{logger.root_dir}/done", pd.DataFrame(), None
    else:
        raise NotImplementedError("This script currently supports only local logging")

    os.makedirs(log_dir, exist_ok=True)
    worker.hyperparameter["dump_id"] = log_dir
    worker.hyperparameter["cache_folder"] = worker.cache_folder
    worker.hyperparameter["result_folder"] = worker.result_folder

    prediction_frame, probing_model = worker.train_run(log_dir=result_log_dir, logger=logger)
    if worker.dump_preds:
        prediction_frame.to_csv(result_log_dir + "/preds.csv")
    worker.mark_run_as_done(logger=logger)
    return f"{logger.root_dir}/done", prediction_frame, probing_model


def evaluate_regression_dataset(
    probing_model: torch.nn.Module,
    dataset: ProbingDataset,
) -> tuple[pd.DataFrame, dict[str, float]]:
    dataloader = probing_model.get_test_dataloader(dataset, 300, shuffle=False)
    all_preds: list[np.ndarray] = []
    all_labels: list[np.ndarray] = []
    all_losses: list[np.ndarray] = []
    all_seen: list[np.ndarray] = []

    probing_model.eval()
    with torch.no_grad():
        for x, y, seen_indices in dataloader:
            x = x.to(probing_model.device)
            y = y.to(probing_model.device)
            pred = probing_model(x).squeeze(1)
            losses = probing_model.loss(pred, y).detach().cpu().numpy()
            all_preds.append(pred.detach().cpu().double().numpy())
            all_labels.append(y.detach().cpu().numpy())
            all_losses.append(losses)
            all_seen.append(seen_indices.detach().cpu().numpy())

    preds = np.concatenate(all_preds) if all_preds else np.array([], dtype=np.float64)
    labels = np.concatenate(all_labels) if all_labels else np.array([], dtype=np.float64)
    losses = np.concatenate(all_losses) if all_losses else np.array([], dtype=np.float64)
    seen = np.concatenate(all_seen) if all_seen else np.array([], dtype=bool)

    instances = [instance_input for instance_input in dataset.inputs]
    prediction_frame = pd.DataFrame(
        {
            "instance": instances,
            "pred": preds,
            "label": labels,
            "loss": losses,
            "seen": np.where(seen, "seen", "ood"),
        }
    )

    if len(preds) == 0:
        metrics = {"ood test error": np.nan, "ood test pearson": np.nan}
    else:
        mse = float(np.mean((preds - labels) ** 2))
        if len(preds) < 2 or np.isclose(np.std(preds), 0.0) or np.isclose(np.std(labels), 0.0):
            pearson = np.nan
        else:
            pearson = float(np.corrcoef(preds, labels)[0, 1])
        metrics = {"ood test error": mse, "ood test pearson": pearson}

    return prediction_frame, metrics


def run_layer(
    layer_idx: int,
    hidden_states: np.ndarray,
    labels: np.ndarray,
    row_ids: list[int],
    permutation_type: str,
    control_task: str,
    reduced_dim: int,
    seed: int,
    fold_idx: int,
    train_pool_idx: np.ndarray,
    test_idx: np.ndarray,
    dev_fraction: float,
    n_total_layers: int,
    results_dir: str,
    model_name: str,
    ood_hidden_states: np.ndarray | None = None,
    ood_labels: np.ndarray | None = None,
    ood_row_ids: list[int] | None = None,
) -> None:
    train_ds, dev_ds, test_ds, ood_ds = make_fold_datasets(
        hidden_states,
        labels,
        row_ids,
        permutation_type,
        control_task,
        reduced_dim,
        train_pool_idx=train_pool_idx,
        test_idx=test_idx,
        dev_fraction=dev_fraction,
        seed=seed,
        ood_hidden_states=ood_hidden_states,
        ood_labels=ood_labels,
        ood_row_ids=ood_row_ids,
    )
    hidden_dim = int(np.asarray(train_ds.inputs_encoded[0]).shape[0])
    probe_name = (
        f"absolute_accuracy_decay__{permutation_type}__"
        f"{control_task.lower()}__L{layer_idx:03d}"
    )

    print(
        f"  {permutation_type:>9} | {control_task:>13} | "
        f"seed={seed} | fold={fold_idx} | layer {layer_idx:03d} | n={len(labels)} | d={hidden_dim}"
    )
    worker = GeneralProbeWorker(
        hyperparameter={
            "seed": seed,
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
            "sample_size": fold_idx,
            "fold": fold_idx,
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
    result_log_dir, _, probing_model = run_worker(worker)

    if ood_ds is not None and probing_model is not None:
        ood_preds, ood_metrics = evaluate_regression_dataset(probing_model, ood_ds)
        ood_preds.to_csv(f"{result_log_dir}/ood_preds.csv")
        with open(f"{result_log_dir}/ood_metrics.json", "w", encoding="utf-8") as f:
            json.dump(ood_metrics, f, indent=2)


def main() -> None:
    args = parse_args()
    internals_dir = Path(args.internals_dir)
    os.makedirs(args.results_dir, exist_ok=True)

    seeds = parse_seeds(args.seeds)
    metadata, layer_files = load_internals(internals_dir)
    n_layers = len(layer_files)
    ood_metadata = None
    if args.ood_internals_dir:
        ood_metadata, ood_layer_files = load_internals(Path(args.ood_internals_dir))
        if layer_files != ood_layer_files:
            raise ValueError("OOD internals dir must contain the same layer_XXX.npy files as the main internals dir")

    print(
        f"Probing {n_layers} layers across "
        f"{metadata['permutation_type'].nunique()} permutation types, "
        f"{len(CONTROL_TASKS)} control settings, "
        f"{len(seeds)} seeds, and {args.num_folds} folds"
    )

    for permutation_type, subset in metadata.groupby("permutation_type", sort=True):
        subset = subset.reset_index(drop=True)
        row_ids = subset["row_id"].astype(int).tolist()
        labels = subset["absolute_accuracy_decay"].to_numpy(dtype=np.float32)
        subset_indices = subset["row_id"].to_numpy(dtype=int)
        ood_subset = None
        if ood_metadata is not None:
            ood_subset = ood_metadata[ood_metadata["permutation_type"] == permutation_type].reset_index(drop=True)

        print(f"\nPermutation type: {permutation_type} | rows={len(subset)}")
        if len(subset) < args.num_folds:
            print(
                f"  Skipping {permutation_type}: requires at least {args.num_folds} rows "
                f"for {args.num_folds}-fold CV, found {len(subset)}"
            )
            continue

        for layer_file in layer_files:
            layer_idx = int(layer_file.replace("layer_", "").replace(".npy", ""))
            layer_states = np.load(internals_dir / layer_file)
            subset_states = layer_states[subset_indices]
            ood_layer_states = ood_labels = ood_row_ids = None
            if ood_subset is not None and not ood_subset.empty:
                full_ood_states = np.load(Path(args.ood_internals_dir) / layer_file)
                ood_indices = ood_subset["row_id"].to_numpy(dtype=int)
                ood_layer_states = full_ood_states[ood_indices]
                ood_labels = ood_subset["absolute_accuracy_decay"].to_numpy(dtype=np.float32)
                ood_row_ids = ood_subset["row_id"].astype(int).tolist()

            for seed in seeds:
                kfold = KFold(n_splits=args.num_folds, shuffle=True, random_state=seed)
                for fold_idx, (train_pool_idx, test_idx) in enumerate(kfold.split(subset_states)):
                    for control_task in CONTROL_TASKS:
                        run_layer(
                            layer_idx=layer_idx,
                            hidden_states=subset_states,
                            labels=labels,
                            row_ids=row_ids,
                            permutation_type=permutation_type,
                            control_task=control_task,
                            reduced_dim=args.reduced_dim,
                            seed=seed,
                            fold_idx=fold_idx,
                            train_pool_idx=train_pool_idx,
                            test_idx=test_idx,
                            dev_fraction=args.dev_fraction,
                            n_total_layers=n_layers,
                            results_dir=args.results_dir,
                            model_name=args.model_name,
                            ood_hidden_states=ood_layer_states,
                            ood_labels=ood_labels,
                            ood_row_ids=ood_row_ids,
                        )

    print(f"\nDone. Results saved to {args.results_dir}/")


if __name__ == "__main__":
    main()
