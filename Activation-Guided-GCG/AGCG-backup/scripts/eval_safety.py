#!/usr/bin/env python3
"""
Evaluate existing completions with substring, LlamaGuard2, and HarmBench.

This script does NOT rerun attacks or regenerate completions; it only
re-scores the JSONs written by activation_pipeline.py.

Usage (from repo root, in an eval-only venv that has vllm + litellm):

  export TOGETHER_API_KEY=...   # for LlamaGuard2
  python eval_safety.py \
      --output-dir outputs/activation_gcg \
      --variants activation_gcg,gcg,ablation \
      --methods substring_matching,llamaguard2,harmbench
"""

import argparse
import json
import os
import sys
from typing import List


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--output-dir",
        default="outputs/activation_gcg",
        help="Directory containing completions/ subdir from activation_pipeline.py",
    )
    p.add_argument(
        "--variants",
        default="baseline,activation_gcg",
        help=(
            "Comma-separated list of variants to evaluate "
            "(include 'baseline' for no-suffix completions). "
            "Add 'gcg' and/or 'ablation' explicitly if you have those outputs."
        ),
    )
    p.add_argument(
        "--methods",
        default="substring_matching,llamaguard2,harmbench",
        help="Comma-separated list of methodologies for evaluate_jailbreak",
    )
    p.add_argument(
        "--split",
        choices=["harmful", "harmless", "both"],
        default="harmful",
        help=(
            "Which completions to evaluate: 'harmful', 'harmless', or 'both'. "
            "Default is 'harmful'."
        ),
    )
    return p.parse_args()


def main():
    args = parse_args()

    # Make refusal_direction code importable
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rd_path = os.path.join(repo_root, "third_party", "refusal_direction")
    if rd_path not in sys.path:
        sys.path.append(rd_path)

    from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak

    methods: List[str] = [m.strip() for m in args.methods.split(",") if m.strip()]
    variants: List[str] = [v.strip() for v in args.variants.split(",") if v.strip()]

    completions_dir = os.path.join(args.output_dir, "completions")

    for variant in variants:
        harm_path = os.path.join(completions_dir, f"{variant}_harmful.json")
        harmless_path = os.path.join(completions_dir, f"{variant}_harmless.json")

        if args.split in ("harmful", "both"):
            if os.path.exists(harm_path):
                print(f"[{variant}] Evaluating harmful completions with {methods} ...")
                evaluate_jailbreak(
                    completions_path=harm_path,
                    methodologies=methods,
                    evaluation_path=os.path.join(args.output_dir, f"{variant}_harmful_eval_safety.json"),
                )
            else:
                print(f"[{variant}] Skipping harmful: missing {harm_path}")

        if args.split in ("harmless", "both"):
            if os.path.exists(harmless_path):
                print(f"[{variant}] Evaluating harmless completions with {methods} ...")
                evaluate_jailbreak(
                    completions_path=harmless_path,
                    methodologies=methods,
                    evaluation_path=os.path.join(args.output_dir, f"{variant}_harmless_eval_safety.json"),
                )
            else:
                print(f"[{variant}] Skipping harmless: missing {harmless_path}")


if __name__ == "__main__":
    main()
