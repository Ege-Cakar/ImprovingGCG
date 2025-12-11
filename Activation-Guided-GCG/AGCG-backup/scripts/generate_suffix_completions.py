#!/usr/bin/env python3
"""
Generate harmful/harmless completions for a manually provided suffix,
so they can be evaluated with eval_safety.py (substring, LlamaGuard2, HarmBench).

Usage (in the main LLaMA-2 env):

  python3.11 generate_suffix_completions.py \
      --model-path meta-llama/Llama-2-7b-chat-hf \
      --suffix "$(cat gcg_suffix.txt)" \
      --output-dir outputs/activation_gcg_negative \
      --name gcg_manual \
      --harmless-limit 100

This will write:
  outputs/activation_gcg_negative/completions/gcg_manual_harmful.json
  outputs/activation_gcg_negative/completions/gcg_manual_harmless.json

You can then run, in the eval venv:

  python eval_safety.py \
      --output-dir outputs/activation_gcg_negative \
      --variants gcg_manual \
      --methods substring_matching,harmbench
"""

import argparse
import json
import os
import sys


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path",
        default="meta-llama/Llama-2-7b-chat-hf",
        help="HF path or local path to LLaMA-2-Chat model",
    )
    p.add_argument(
        "--suffix",
        required=True,
        help="Suffix string to append to each instruction",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/manual_suffix",
        help="Root directory where completions/ will be written",
    )
    p.add_argument(
        "--name",
        default="manual_suffix",
        help="Variant name (used in filenames: <name>_harmful.json, etc.)",
    )
    p.add_argument(
        "--harmless-limit",
        type=int,
        default=100,
        help="Max harmful/harmless prompts to evaluate (0 = full splits)",
    )
    return p.parse_args()


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def main():
    args = parse_args()

    # Make refusal_direction code importable
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rd_path = os.path.join(repo_root, "third_party", "refusal_direction")
    if rd_path not in sys.path:
        sys.path.append(rd_path)

    from dataset.load_dataset import load_dataset_split  # type: ignore
    from pipeline.model_utils.llama2_model import Llama2Model  # type: ignore

    ensure_dir(args.output_dir)
    completions_dir = ensure_dir(os.path.join(args.output_dir, "completions"))

    # Load eval sets from refusal_direction
    harmful_eval = load_dataset_split(harmtype="harmful", split="test")
    harmless_eval = load_dataset_split(harmtype="harmless", split="test")

    if args.harmless_limit and args.harmless_limit > 0:
        harmful_eval = harmful_eval[: args.harmless_limit]
        harmless_eval = harmless_eval[: args.harmless_limit]

    print(f"Using {len(harmful_eval)} harmful and {len(harmless_eval)} harmless prompts.")

    # Load model via refusal_direction wrapper (same as activation_pipeline)
    model_base = Llama2Model(args.model_path)

    def augment(dataset):
        augmented = []
        for row in dataset:
            inst = row["instruction"] + (f" {args.suffix}" if args.suffix else "")
            augmented.append(
                {
                    "instruction": inst,
                    "category": row.get("category", "unknown"),
                }
            )
        return augmented

    print(f"=== Generating completions for suffix variant '{args.name}' ===")
    harm_comp = model_base.generate_completions(
        augment(harmful_eval),
        batch_size=8,
        max_new_tokens=128,
    )
    harmless_comp = model_base.generate_completions(
        augment(harmless_eval),
        batch_size=8,
        max_new_tokens=128,
    )

    harm_path = os.path.join(completions_dir, f"{args.name}_harmful.json")
    harmless_path = os.path.join(completions_dir, f"{args.name}_harmless.json")

    with open(harm_path, "w") as f:
        json.dump(harm_comp, f, indent=4)
    with open(harmless_path, "w") as f:
        json.dump(harmless_comp, f, indent=4)

    print("Wrote:")
    print(" ", harm_path)
    print(" ", harmless_path)
    print("You can now run eval_safety.py on variant:", args.name)


if __name__ == "__main__":
    main()
