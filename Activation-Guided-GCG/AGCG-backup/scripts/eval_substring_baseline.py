#!/usr/bin/env python3
"""
Compute substring-based ASR for a no-suffix baseline.

This script:
  - Uses the refusal_direction harmful/harmless *test* splits
  - Generates completions with Llama-2-7B-Chat (no suffix, no ablation)
  - Evaluates substring matching ASR on harmful and harmless completions

Usage:
  python eval_substring_baseline.py \
      --model-path meta-llama/Llama-2-7b-chat-hf \
      --output-dir outputs/substring_baseline \
      --harmless-limit 100
"""

import argparse
import json
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from fastchat.model import get_conversation_template


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--model-path",
        default="meta-llama/Llama-2-7b-chat-hf",
        help="HF path or local path to LLaMA-2-Chat model",
    )
    p.add_argument(
        "--output-dir",
        default="outputs/substring_baseline",
        help="Where to write completions and eval JSONs",
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

    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    rd_path = os.path.join(repo_root, "third_party", "refusal_direction")
    if rd_path not in sys.path:
        sys.path.append(rd_path)

    from dataset.load_dataset import load_dataset_split  # type: ignore
    from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak  # type: ignore

    torch.set_grad_enabled(False)

    ensure_dir(args.output_dir)
    completions_dir = ensure_dir(os.path.join(args.output_dir, "completions"))

    # Load eval sets from refusal_direction
    harmful_eval = load_dataset_split(harmtype="harmful", split="test")
    harmless_eval = load_dataset_split(harmtype="harmless", split="test")

    if args.harmless_limit and args.harmless_limit > 0:
        harmful_eval = harmful_eval[: args.harmless_limit]
        harmless_eval = harmless_eval[: args.harmless_limit]

    print(f"Using {len(harmful_eval)} harmful and {len(harmless_eval)} harmless prompts.")

    # Load plain HF model/tokenizer (no ModelBase, so no jaxtyping)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype=torch.float16,
        device_map="auto" if torch.cuda.is_available() else None,
    )
    model.to(device)
    model.eval()

    def build_prompt(user_content: str) -> str:
        from fastchat.model import get_conversation_template

        conv = get_conversation_template("llama-2")
        conv.append_message(conv.roles[0], user_content)
        conv.append_message(conv.roles[1], None)
        return conv.get_prompt()

    def generate_batch(dataset):
        completions = []
        gen_cfg = GenerationConfig(
            max_new_tokens=128,
            do_sample=False,
            pad_token_id=tok.pad_token_id,
        )
        for row in dataset:
            prompt = build_prompt(row["instruction"])
            inputs = tok(prompt, return_tensors="pt").to(model.device)
            with torch.no_grad():
                output_ids = model.generate(**inputs, generation_config=gen_cfg)
            gen = output_ids[0, inputs.input_ids.shape[1] :]
            resp = tok.decode(gen, skip_special_tokens=True).strip()
            completions.append(
                {
                    "category": row.get("category", "unknown"),
                    "prompt": row["instruction"],
                    "response": resp,
                }
            )
        return completions

    # Generate baseline completions
    print("=== Generating baseline completions (no suffix) ===")
    harm_comp = generate_batch(harmful_eval)
    harmless_comp = generate_batch(harmless_eval)

    harm_path = os.path.join(completions_dir, "baseline_harmful.json")
    harmless_path = os.path.join(completions_dir, "baseline_harmless.json")

    with open(harm_path, "w") as f:
        json.dump(harm_comp, f, indent=4)
    with open(harmless_path, "w") as f:
        json.dump(harmless_comp, f, indent=4)

    # Substring ASR evaluation
    methods = ["substring_matching"]
    print("=== Evaluating substring ASR for baseline (harmful) ===")
    eval_harm = evaluate_jailbreak(
        completions=harm_comp,
        methodologies=methods,
        evaluation_path=os.path.join(args.output_dir, "baseline_harmful_eval_substring.json"),
    )
    print("Baseline substring ASR (harmful):", eval_harm.get("substring_matching_success_rate", "n/a"))

    print("=== Evaluating substring ASR for baseline (harmless) ===")
    eval_harmless = evaluate_jailbreak(
        completions=harmless_comp,
        methodologies=methods,
        evaluation_path=os.path.join(args.output_dir, "baseline_harmless_eval_substring.json"),
    )
    print("Baseline substring ASR (harmless):", eval_harmless.get("substring_matching_success_rate", "n/a"))

    print("Done. Artifacts in", args.output_dir)


if __name__ == "__main__":
    main()
