"""
Extract eval-adoption internals using a fast forward-only pass.

This script only supports one representation:

- `input_last_token`: the final prompt-token hidden state from each layer

Duplicate prompts are encoded once and then expanded back to the full row set.
The emitted directory contains `metadata.csv` plus `layer_XXX.npy` files for
direct use by `scripts/probe.py`.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

REPO_ROOT = Path(__file__).resolve().parents[1]
SYSTEM_PROMPT_PATH = REPO_ROOT / "prompts" / "solve.txt"
OPTIONAL_METADATA_COLUMNS = ["is_robust", "model_is_robust"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-csv",
        required=True,
        help="Path to eval-adoption dataset CSV.",
    )
    parser.add_argument(
        "--model-id",
        required=True,
        help="HF model id used to extract hidden states.",
    )
    parser.add_argument(
        "--output-dir",
        default="data/eval_adoption_internals",
        help="Directory to write metadata.csv and layer_XXX.npy files.",
    )
    parser.add_argument(
        "--device",
        default="auto",
        choices=["auto", "cpu", "cuda"],
        help="Device to run extraction on.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> str:
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("Requested --device cuda, but CUDA is not available.")
    return device_arg


def build_prompt(tokenizer, user_text: str, system_prompt: str) -> str:
    if hasattr(tokenizer, "apply_chat_template"):
        try:
            return tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_text},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        except Exception:
            pass
    return f"{system_prompt}\n\n{user_text}"


def extract_hidden_states(
    df: pd.DataFrame,
    model,
    tokenizer,
    system_prompt: str,
    output_dir: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    prompt_cache: dict[str, list[np.ndarray]] = {}
    layer_storage: list[list[np.ndarray]] | None = None

    unique_rows = df.drop_duplicates(subset=["original_problem"], keep="first").reset_index(drop=True)
    print(
        f"Extracting input_last_token internals for {len(df)} rows "
        f"({len(unique_rows)} unique prompts) ..."
    )

    for idx, row in unique_rows.iterrows():
        print(
            f"  [{idx + 1:>{len(str(len(unique_rows)))}}/{len(unique_rows)}]  "
            f"row_id={int(row['row_id'])}, problem_id={row['problem_id']!r}, model_id={row['model_id']!r}"
        )
        prompt = build_prompt(tokenizer, row["original_problem"], system_prompt)
        enc = tokenizer(prompt, return_tensors="pt", truncation=True)
        enc = {key: value.to(model.device) for key, value in enc.items()}

        with torch.no_grad():
            output = model(
                **enc,
                output_hidden_states=True,
                return_dict=True,
                use_cache=False,
            )

        hidden_states = output.hidden_states
        if hidden_states is None:
            raise RuntimeError("Model forward pass returned no hidden states.")

        if layer_storage is None:
            n_layers = len(hidden_states)
            hidden_dim = hidden_states[0].shape[-1]
            layer_storage = [[] for _ in range(n_layers)]
            print(f"Layers: {n_layers} | hidden_dim: {hidden_dim}")

        prompt_cache[row["original_problem"]] = [
            hidden_state[0, -1, :].detach().float().cpu().numpy()
            for hidden_state in hidden_states
        ]

    if layer_storage is None:
        raise RuntimeError("No hidden states were extracted.")

    metadata_rows: list[dict] = []
    for _, row in df.iterrows():
        cached = prompt_cache[row["original_problem"]]
        for layer_idx, vec in enumerate(cached):
            layer_storage[layer_idx].append(vec)

        metadata_rows.append(
            {
                "row_id": int(row["row_id"]),
                "problem_id": row["problem_id"],
                "model_id": row["model_id"],
                "dataset_id": row["dataset_id"],
                "permutation_type": row["permutation_type"],
                "absolute_accuracy_decay": float(row["absolute_accuracy_decay"]),
                "original_problem": row["original_problem"],
                "extraction_view": "input_last_token",
                **{
                    col: row[col]
                    for col in OPTIONAL_METADATA_COLUMNS
                    if col in row and pd.notna(row[col])
                },
            }
        )

    for layer_idx, rows in enumerate(layer_storage):
        np.save(output_dir / f"layer_{layer_idx:03d}.npy", np.stack(rows))

    metadata = pd.DataFrame(metadata_rows).sort_values("row_id")
    metadata.to_csv(output_dir / "metadata.csv", index=False)


def main() -> None:
    args = parse_args()
    output_dir = Path(args.output_dir)
    device = resolve_device(args.device)
    dtype = torch.bfloat16 if device == "cuda" and torch.cuda.is_bf16_supported() else (
        torch.float16 if device == "cuda" else torch.float32
    )

    system_prompt = SYSTEM_PROMPT_PATH.read_text().strip()
    df = pd.read_csv(args.dataset_csv).reset_index(drop=True)
    df["row_id"] = df.index
    df["absolute_accuracy_decay"] = df["absolute_accuracy_decay"].astype(float)

    print(f"Loading model: {args.model_id}")
    print(f"Using device: {device} | dtype: {dtype}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForCausalLM.from_pretrained(args.model_id, torch_dtype=dtype)
    model.to(device)
    model.eval()

    extract_hidden_states(
        df=df,
        model=model,
        tokenizer=tokenizer,
        system_prompt=system_prompt,
        output_dir=output_dir,
    )
    print(f"Saved internals to: {output_dir}")


if __name__ == "__main__":
    main()
