#!/usr/bin/env python3
"""
Generate harmful/harmless completions for:
  - each optimized suffix in final_analysis/method_suffix_summary.csv
  - a no-suffix baseline
  - refusal-direction ablation (no suffix, direction removed via hooks)

Outputs are written incrementally under --output-dir (default: final_analysis/generated_completions).
Files:
  <output-dir>/completions/<variant>_harmful.json
  <output-dir>/completions/<variant>_harmless.json

Prereqs:
  - Works with Llama-2 style models using the refusal_direction Llama2Model wrapper.
  - Direction artifacts available (direction.pt + direction_metadata.json).

Example:
  python final_analysis/generate_all_completions.py \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --direction-path third_party/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction.pt \
    --direction-meta third_party/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction_metadata.json
"""

import argparse
import csv
import json
import os
import sys
from typing import List, Dict, Tuple

# Make vendored modules importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD_PATH = os.path.join(REPO_ROOT, "third_party", "refusal_direction")
if RD_PATH not in sys.path:
    sys.path.append(RD_PATH)

from dataset.load_dataset import load_dataset_split  # type: ignore
from pipeline.model_utils.llama2_model import Llama2Model  # type: ignore
from pipeline.utils.hook_utils import get_all_direction_ablation_hooks  # type: ignore


def load_direction(direction_path: str, meta_path: str) -> Tuple:
    direction = torch.load(direction_path, map_location="cpu")
    with open(meta_path) as f:
        meta = json.load(f)
    return direction, meta["layer"], meta["pos"]


def read_suffixes(csv_path: str) -> List[Tuple[str, str]]:
    variants = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            method = row["method"].strip()
            suffix = row["suffix"]
            variants.append((method, suffix))
    return variants


def ensure_dir(path: str) -> str:
    os.makedirs(path, exist_ok=True)
    return path


def generate_for_variant(
    model_base,
    variant_name: str,
    suffix: str,
    harmful: List[Dict],
    harmless: List[Dict],
    out_dir: str,
    fwd_pre_hooks=None,
    fwd_hooks=None,
):
    """Generate completions for one variant and write to disk."""
    fwd_pre_hooks = fwd_pre_hooks or []
    fwd_hooks = fwd_hooks or []

    def augment(dataset):
        augmented = []
        for row in dataset:
            inst = row["instruction"] + (f" {suffix}" if suffix else "")
            augmented.append({"instruction": inst, "category": row.get("category", "unknown")})
        return augmented

    completions_dir = ensure_dir(os.path.join(out_dir, "completions"))
    harm_path = os.path.join(completions_dir, f"{variant_name}_harmful.json")
    harmless_path = os.path.join(completions_dir, f"{variant_name}_harmless.json")

    print(f"[{variant_name}] Generating harmful completions ...")
    harm_comp = model_base.generate_completions(
        augment(harmful), fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=8, max_new_tokens=128
    )
    json.dump(harm_comp, open(harm_path, "w"), indent=4)
    print(f"[{variant_name}] Wrote {harm_path}")

    print(f"[{variant_name}] Generating harmless completions ...")
    harmless_comp = model_base.generate_completions(
        augment(harmless), fwd_pre_hooks=fwd_pre_hooks, fwd_hooks=fwd_hooks, batch_size=8, max_new_tokens=128
    )
    json.dump(harmless_comp, open(harmless_path, "w"), indent=4)
    print(f"[{variant_name}] Wrote {harmless_path}")


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True, help="HF path or local path to Llama-2 chat model")
    ap.add_argument(
        "--direction-path",
        default="third_party/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction.pt",
        help="Path to refusal direction .pt",
    )
    ap.add_argument(
        "--direction-meta",
        default="third_party/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction_metadata.json",
        help="Path to JSON with {layer,pos}",
    )
    ap.add_argument(
        "--suffix-csv",
        default=os.path.join(REPO_ROOT, "final_analysis", "method_suffix_summary.csv"),
        help="CSV with columns method,loss,suffix",
    )
    ap.add_argument(
        "--output-dir",
        default=os.path.join(REPO_ROOT, "final_analysis", "generated_completions"),
        help="Where to write completions/",
    )
    ap.add_argument(
        "--harmless-limit",
        type=int,
        default=0,
        help="Limit for harmful/harmless prompts (0 = use full test splits)",
    )
    return ap.parse_args()


if __name__ == "__main__":
    import torch

    args = parse_args()
    ensure_dir(args.output_dir)

    direction, layer, pos = torch.load(args.direction_path, map_location="cpu"), None, None
    # Load layer/pos metadata
    with open(args.direction_meta) as f:
        meta = json.load(f)
    layer, pos = meta["layer"], meta["pos"]

    # Model
    model_base = Llama2Model(args.model_path)

    # Data
    harmful_eval = load_dataset_split(harmtype="harmful", split="test")
    harmless_eval = load_dataset_split(harmtype="harmless", split="test")
    if args.harmless_limit and args.harmless_limit > 0:
        harmful_eval = harmful_eval[: args.harmless_limit]
        harmless_eval = harmless_eval[: args.harmless_limit]

    # Hooks for ablation
    ablation_pre, ablation_hooks = get_all_direction_ablation_hooks(model_base, direction)

    # Variants: baseline, ablation, and suffixes from CSV
    variants = [("baseline", None, [], []), ("ablation", None, ablation_pre, ablation_hooks)]

    for method, suffix in read_suffixes(args.suffix_csv):
        variants.append((method, suffix, [], []))

    for name, suffix, pre_hooks, hooks in variants:
        generate_for_variant(
            model_base=model_base,
            variant_name=name,
            suffix=suffix or "",
            harmful=harmful_eval,
            harmless=harmless_eval,
            out_dir=args.output_dir,
            fwd_pre_hooks=pre_hooks,
            fwd_hooks=hooks,
        )

    print("All completions generated under", args.output_dir)
