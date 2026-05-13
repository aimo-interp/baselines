"""
Extract layer-wise hidden states for eval-adoption CSV rows.

This reads `dataset_as_table.csv`, runs a causal LM over each `original_problem`,
and saves one `.npy` file per layer containing the last input-token hidden state.
The emitted metadata is aligned row-for-row with the saved arrays, so downstream
probing can subset by `permutation_type` and regress on `absolute_accuracy_decay`.

Example:
    ./.venv/bin/python 04_extract_eval_adoption_internals.py \
        --dataset-csv /Users/tresi/Projects/eval-adoption/dataset_as_table.csv \
        --model-id Qwen/Qwen2.5-0.5B-Instruct \
        --output-dir data/eval_adoption_internals
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

import lamina
from lamina import InternalsDataset, InternalsInstance

SYSTEM_PROMPT_PATH = "prompts/solve.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-csv",
        required=True,
        help="Path to eval-adoption dataset_as_table.csv",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="HF model id used to extract hidden states",
    )
    parser.add_argument(
        "--output-dir",
        default="data/eval_adoption_internals",
        help="Directory to write metadata.csv and layer_XXX.npy files",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run extraction on",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    return device_arg


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device == "cuda" else torch.float32

    system_prompt = Path(SYSTEM_PROMPT_PATH).read_text().strip()
    df = pd.read_csv(args.dataset_csv).reset_index(drop=True)
    df["row_id"] = df.index
    df["absolute_accuracy_decay"] = df["absolute_accuracy_decay"].astype(float)

    print(f"Loading model: {args.model_id}")
    print(f"Using device: {device} | dtype: {dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()

    instances = [
        InternalsInstance(
            text=row["original_problem"],
            properties={
                "row_id": int(row["row_id"]),
                "problem_id": row["problem_id"],
                "model_id": row["model_id"],
                "dataset_id": row["dataset_id"],
                "permutation_type": row["permutation_type"],
                "absolute_accuracy_decay": float(row["absolute_accuracy_decay"]),
            },
            system_prompt=system_prompt,
        )
        for _, row in df.iterrows()
    ]
    dataset = InternalsDataset(instances)

    print(f"Extracting internals for {len(instances)} rows ...")
    records = dataset.run(model, tokenizer, generate_kwargs={"max_new_tokens": 1})

    n_layers = records[0].run.num_layers
    hidden_dim = records[0].run.input_hidden_states[0].shape[-1]
    print(f"Layers: {n_layers}  |  hidden_dim: {hidden_dim}")

    for layer_idx in range(n_layers):
        layer_vecs = np.stack(
            [rec.run.input_hidden_states[layer_idx][0, -1, :] for rec in records]
        )
        np.save(output_dir / f"layer_{layer_idx:03d}.npy", layer_vecs)

    metadata = pd.DataFrame(
        [
            {
                "row_id": rec.properties["row_id"],
                "problem_id": rec.properties["problem_id"],
                "model_id": rec.properties["model_id"],
                "dataset_id": rec.properties["dataset_id"],
                "permutation_type": rec.properties["permutation_type"],
                "absolute_accuracy_decay": rec.properties["absolute_accuracy_decay"],
                "original_problem": rec.instance.text,
            }
            for rec in records
        ]
    ).sort_values("row_id")
    metadata.to_csv(output_dir / "metadata.csv", index=False)

    print(f"Saved metadata and {n_layers} layer files to {output_dir}")


if __name__ == "__main__":
    main()
