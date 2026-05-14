"""
Extract eval-adoption internals for multiple Lamina-supported views.

For each CSV row, this script runs the model through Lamina and saves one
internals directory per requested representation view:

- `input_last_token`: last prompt token hidden state
- `last_thinking_token`: last generated token hidden state
- `output_last_token`: alias for the last generated token hidden state
- `average_output`: mean hidden state across generated tokens

Each emitted directory contains `metadata.csv` plus `layer_XXX.npy` files, so
the downstream probing script can be pointed at any one of the views directly.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from lamina import InternalsDataset, InternalsInstance

SYSTEM_PROMPT_PATH = "prompts/solve.txt"
DEFAULT_VIEWS = [
    "input_last_token",
    "last_thinking_token",
    "output_last_token",
    "average_output",
]


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
        help="Base directory to write metadata.csv and layer_XXX.npy files",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run extraction on",
    )
    parser.add_argument(
        "--views",
        default=",".join(DEFAULT_VIEWS),
        help=(
            "Comma-separated extraction views. Supported values: "
            "input_last_token,last_thinking_token,output_last_token,average_output"
        ),
    )
    parser.add_argument(
        "--max-new-tokens",
        type=int,
        default=256,
        help="Maximum number of generated tokens captured for output-based views.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    return device_arg


def parse_views(views_arg: str) -> list[str]:
    supported = set(DEFAULT_VIEWS)
    views = [chunk.strip() for chunk in views_arg.split(",") if chunk.strip()]
    if not views:
        raise ValueError("At least one extraction view must be provided.")
    invalid = sorted(set(views) - supported)
    if invalid:
        raise ValueError(f"Unsupported extraction views: {invalid}")
    return views


def view_output_dir(base_dir: Path, view: str) -> Path:
    return base_dir / view


def build_instances(df: pd.DataFrame, system_prompt: str) -> list[InternalsInstance]:
    return [
        InternalsInstance(
            text=row["original_problem"],
            properties={
                "row_id": int(row["row_id"]),
                "problem_id": row["problem_id"],
                "model_id": row["model_id"],
                "dataset_id": row["dataset_id"],
                "permutation_type": row["permutation_type"],
                "absolute_accuracy_decay": float(row["absolute_accuracy_decay"]),
                "original_problem": row["original_problem"],
            },
            system_prompt=system_prompt,
        )
        for _, row in df.iterrows()
    ]


def extract_view_vector(record, view: str, layer_idx: int) -> np.ndarray:
    run = record.run
    if view == "input_last_token":
        if run.input_hidden_states is None:
            raise RuntimeError("Lamina record has no input_hidden_states.")
        return np.asarray(run.input_hidden_states[layer_idx][0, -1, :], dtype=np.float32)

    if view in {"last_thinking_token", "output_last_token"}:
        if run.output_hidden_states is None:
            raise RuntimeError("Lamina record has no output_hidden_states.")
        layer_output = run.output_hidden_states[layer_idx]
        if layer_output.shape[1] == 0:
            raise RuntimeError("No generated tokens were captured; increase --max-new-tokens.")
        # Lamina exposes generated-token hidden states. We treat the final
        # generated token representation as both the "last thinking token"
        # view and the explicit "output last token" view.
        return np.asarray(layer_output[0, -1, :], dtype=np.float32)

    if view == "average_output":
        if run.output_hidden_states_mean is None:
            raise RuntimeError("Lamina record has no output_hidden_states_mean.")
        return np.asarray(run.output_hidden_states_mean[layer_idx, 0, :], dtype=np.float32)

    raise ValueError(f"Unknown extraction view: {view}")


def save_view(
    records: list,
    metadata_rows: list[dict],
    output_dir: Path,
    view: str,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    first_run = records[0].run
    if view == "input_last_token":
        source = first_run.input_hidden_states
    elif view in {"last_thinking_token", "output_last_token"}:
        source = first_run.output_hidden_states
    else:
        source = first_run.output_hidden_states

    if source is None:
        raise RuntimeError(f"Cannot save view {view!r}: source hidden states are missing.")

    n_layers = len(source)
    for layer_idx in range(n_layers):
        layer_vecs = np.stack([
            extract_view_vector(record, view, layer_idx)
            for record in records
        ])
        np.save(output_dir / f"layer_{layer_idx:03d}.npy", layer_vecs)

    metadata = pd.DataFrame(metadata_rows).sort_values("row_id")
    metadata.to_csv(output_dir / "metadata.csv", index=False)


def main() -> None:
    args = parse_args()
    base_output_dir = Path(args.output_dir)
    views = parse_views(args.views)
    device = resolve_device(args.device)
    # Lamina's worker converts captured tensors through NumPy. On many setups,
    # direct NumPy conversion for bfloat16 tensors fails, so prefer float16 on
    # CUDA here and keep float32 on CPU.
    dtype = torch.float16 if device == "cuda" else torch.float32

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

    instances = build_instances(df, system_prompt)
    dataset = InternalsDataset(instances)

    print(f"Extracting internals for {len(instances)} rows ...")
    records = dataset.run(
        model,
        tokenizer,
        generate_kwargs={"max_new_tokens": args.max_new_tokens},
        verbose=True,
    )
    if not records:
        raise RuntimeError("Lamina returned no records.")

    first_run = records[0].run
    input_layers = len(first_run.input_hidden_states or [])
    output_layers = len(first_run.output_hidden_states or [])
    hidden_dim = None
    if first_run.input_hidden_states is not None:
        hidden_dim = first_run.input_hidden_states[0].shape[-1]
    elif first_run.output_hidden_states is not None:
        hidden_dim = first_run.output_hidden_states[0].shape[-1]
    print(
        f"Input layers: {input_layers} | Output layers: {output_layers} | hidden_dim: {hidden_dim}"
    )

    base_metadata_rows = [
        {
            "row_id": int(record.properties["row_id"]),
            "problem_id": record.properties["problem_id"],
            "model_id": record.properties["model_id"],
            "dataset_id": record.properties["dataset_id"],
            "permutation_type": record.properties["permutation_type"],
            "absolute_accuracy_decay": float(record.properties["absolute_accuracy_decay"]),
            "original_problem": record.properties["original_problem"],
        }
        for record in records
    ]

    for view in views:
        print(f"Saving view: {view}")
        view_metadata_rows = [
            {**row, "extraction_view": view}
            for row in base_metadata_rows
        ]
        save_view(records, view_metadata_rows, view_output_dir(base_output_dir, view), view)

    print(
        "Saved view directories: "
        + ", ".join(str(view_output_dir(base_output_dir, view)) for view in views)
    )


if __name__ == "__main__":
    main()
