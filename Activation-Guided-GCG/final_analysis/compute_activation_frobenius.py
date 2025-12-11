#!/usr/bin/env python3
"""
Compute both cosine similarity and Frobenius distance between suffix-induced
activations and their refusal-direction ablated counterparts across all
layers/tokens.

For each example, we flatten H tensors and compute:
  cos = <H_suffix, H_suffix_ablation> /
        (||H_suffix|| * ||H_suffix_ablation||)
  dist = ||H_suffix - H_suffix_ablation||_F
where H_* are stacked resid_pre activations (layers x seq x d) captured during a
single forward pass. Sequences are truncated to the minimum length within each
pair (variant vs variant_ablation) for a fair comparison.

Outputs:
  final_analysis/activation_distances.json (by default) with per-method stats:
    mean_cosine, median_cosine, cosine_values (per example),
    mean_distance, median_distance, distances (per example).

Example:
  python final_analysis/compute_activation_frobenius.py \
    --model-path meta-llama/Llama-2-7b-chat-hf \
    --direction-path third_party/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction.pt \
    --direction-meta third_party/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction_metadata.json \
    --suffix-csv final_analysis/method_suffix_summary.csv \
    --n-examples 8
"""

import argparse
import csv
import json
import os
import sys
from typing import Dict, List, Tuple

import torch

# Make vendored modules importable
REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RD_PATH = os.path.join(REPO_ROOT, "third_party", "refusal_direction")
if RD_PATH not in sys.path:
    sys.path.append(RD_PATH)

from dataset.load_dataset import load_dataset_split  # type: ignore
from pipeline.model_utils.llama2_model import Llama2Model  # type: ignore
from pipeline.utils.hook_utils import get_direction_ablation_input_pre_hook  # type: ignore


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model-path", required=True)
    ap.add_argument(
        "--direction-path",
        default="third_party/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction.pt",
    )
    ap.add_argument(
        "--direction-meta",
        default="third_party/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction_metadata.json",
    )
    ap.add_argument(
        "--suffix-csv",
        default=os.path.join(REPO_ROOT, "final_analysis", "method_suffix_summary.csv"),
        help="CSV with columns method,loss,suffix",
    )
    ap.add_argument("--output", default=os.path.join(REPO_ROOT, "final_analysis", "activation_distances.json"))
    ap.add_argument("--n-examples", type=int, default=4, help="Number of harmful test prompts to sample")
    ap.add_argument("--device", default="cuda:0")
    return ap.parse_args()


def load_direction(direction_path: str, meta_path: str):
    direction = torch.load(direction_path, map_location="cpu")
    with open(meta_path) as f:
        meta = json.load(f)
    return direction, meta["layer"], meta["pos"]


def read_suffixes(csv_path: str) -> List[Tuple[str, str]]:
    variants = []
    with open(csv_path, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            variants.append((row["method"].strip(), row["suffix"]))
    return variants


def build_inputs(model_base, instruction: str, suffix: str = ""):
    text = instruction if not suffix else f"{instruction} {suffix}"
    tok = model_base.tokenize_instructions_fn(instructions=[text])
    return {k: v.to(model_base.model.device) for k, v in tok.items()}


def capture_resid_pre(model_base, inputs, pre_hooks=None):
    """Run a forward pass and capture resid_pre at every block (list of [seq, d])."""
    pre_hooks = pre_hooks or []
    captured: List[torch.Tensor] = []

    handles = []

    def hook_fn(module, hook_input):
        # hook_input[0]: [batch, seq, d_model]
        captured.append(hook_input[0].detach().to(torch.float32).cpu()[0])

    # Register our capture hooks
    for blk in model_base.model_block_modules:
        handles.append(blk.register_forward_pre_hook(hook_fn))
    # Register any extra hooks (e.g., ablation)
    for module, fn in pre_hooks:
        handles.append(module.register_forward_pre_hook(fn))

    with torch.no_grad():
        _ = model_base.model(**inputs)

    for h in handles:
        h.remove()

    return captured  # list length = num_layers, each [seq, d]


def stack_layers(layer_acts: List[torch.Tensor]) -> torch.Tensor:
    max_seq = min(act.shape[0] for act in layer_acts)
    trimmed = [act[:max_seq] for act in layer_acts]
    return torch.stack(trimmed, dim=0)  # [layers, seq, d]


def cosine_similarity(a: torch.Tensor, b: torch.Tensor) -> float:
    v1 = a.reshape(-1)
    v2 = b.reshape(-1)
    # Guard against zero vectors
    n1 = torch.linalg.norm(v1)
    n2 = torch.linalg.norm(v2)
    denom = (n1 * n2).item()
    if denom == 0.0:
        return 0.0
    return float(torch.dot(v1, v2).item() / denom)


def main():
    args = parse_args()
    torch.set_grad_enabled(False)

    direction, layer, pos = load_direction(args.direction_path, args.direction_meta)

    model_base = Llama2Model(args.model_path)
    model_base.model.to(args.device)

    # Data
    harmful = load_dataset_split(harmtype="harmful", split="test")
    harmful = harmful[: args.n_examples]

    # Ablation hooks: apply to every layer
    ablation_pre = [(blk, get_direction_ablation_input_pre_hook(direction=direction)) for blk in model_base.model_block_modules]

    variants = [("baseline", "")]
    variants += read_suffixes(args.suffix_csv)

    results: Dict[str, Dict] = {}

    for method, suffix in variants:
        print(f"Evaluating {method} ...")
        sims = []
        dists = []
        for row in harmful:
            instr = row["instruction"]

            # Variant inputs (with suffix) and baseline inputs (no suffix)
            inp_var = build_inputs(model_base, instr, suffix=suffix)

            # Capture activations with/without ablation for both variant and baseline
            acts_var = capture_resid_pre(model_base, inp_var, pre_hooks=[])
            acts_var_abl = capture_resid_pre(model_base, inp_var, pre_hooks=ablation_pre)

            # Align layers
            min_layers = min(len(acts_var), len(acts_var_abl))
            acts_var = acts_var[:min_layers]
            acts_var_abl = acts_var_abl[:min_layers]

            H_var = stack_layers(acts_var)
            H_var_abl = stack_layers(acts_var_abl)

            # Align sequence length within the pair
            min_seq_var = min(H_var.shape[1], H_var_abl.shape[1])
            H_var = H_var[:, :min_seq_var]
            H_var_abl = H_var_abl[:, :min_seq_var]

            sim = cosine_similarity(H_var, H_var_abl)
            dist = torch.linalg.norm((H_var - H_var_abl).reshape(-1), ord=2).item()
            sims.append(sim)
            dists.append(dist)

        results[method] = {
            "n_examples": len(harmful),
            "mean_cosine": sum(sims) / len(sims) if sims else None,
            "median_cosine": sorted(sims)[len(sims) // 2] if sims else None,
            "cosine_values": sims,
            "mean_distance": sum(dists) / len(dists) if dists else None,
            "median_distance": sorted(dists)[len(dists) // 2] if dists else None,
            "distances": dists,
        }

    with open(args.output, "w") as f:
        json.dump(results, f, indent=4)
    print("Wrote", args.output)


if __name__ == "__main__":
    main()
