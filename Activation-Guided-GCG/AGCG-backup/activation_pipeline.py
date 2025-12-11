"""
Unified CLI to run activation-based GCG (refusal-direction objective),
standard GCG, and refusal-direction ablation baselines on Llama-2-7B-Chat.

Outputs:
  - Optimized suffixes for activation-GCG and standard GCG (optional)
  - Completions for harmful/harmless datasets
  - Jailbreak ASR metrics (substring + optional LlamaGuard2)
  - CE/perplexity metrics on harmless data
Artifacts are written under --output-dir.
"""
import argparse
import json
import os
import random
import sys
from types import SimpleNamespace

import torch
import torch.multiprocessing as mp

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))
THIRD_PARTY_DIR = os.path.join(REPO_ROOT, "third_party")

# Make vendored libraries importable
sys.path.append(os.path.join(THIRD_PARTY_DIR, "gcg"))
sys.path.append(os.path.join(THIRD_PARTY_DIR, "refusal_direction"))

from llm_attacks import get_workers
from llm_attacks.activation_gcg import (
    AttackPrompt as ActAttackPrompt,
    PromptManager as ActPromptManager,
    MultiPromptAttack as ActMultiPromptAttack,
)
from llm_attacks.gcg import AttackPrompt as GcgAttackPrompt
from llm_attacks.gcg import PromptManager as GcgPromptManager
from llm_attacks.gcg import MultiPromptAttack as GcgMultiPromptAttack

from dataset.load_dataset import load_dataset_split
from pipeline.model_utils.llama2_model import Llama2Model
from pipeline.utils.hook_utils import (
    get_all_direction_ablation_hooks,
    get_direction_ablation_input_pre_hook,
)
from pipeline.submodules.evaluate_jailbreak import evaluate_jailbreak
from pipeline.submodules.evaluate_loss import evaluate_loss


LLAMA2_DIRECTION_PATH = os.path.join(
    "third_party", "refusal_direction", "pipeline", "runs", "llama-2-7b-chat-hf", "direction.pt"
)
LLAMA2_DIRECTION_META = os.path.join(
    "third_party", "refusal_direction", "pipeline", "runs", "llama-2-7b-chat-hf", "direction_metadata.json"
)


def parse_args():
    p = argparse.ArgumentParser(description="Activation-based GCG pipeline (Llama-2-7B-Chat)")
    p.add_argument("--model-path", default="meta-llama/Llama-2-7b-chat-hf")
    p.add_argument(
        "--direction-path",
        default=LLAMA2_DIRECTION_PATH,
        help="Precomputed refusal direction .pt file",
    )
    p.add_argument(
        "--direction-meta",
        default=LLAMA2_DIRECTION_META,
        help="JSON with {layer,pos}",
    )
    p.add_argument("--output-dir", default="outputs/activation_gcg")
    p.add_argument("--n-train", type=int, default=25, help="Number of harmful train prompts")
    p.add_argument("--n-steps", type=int, default=200, help="Optimization steps for activation GCG")
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--topk", type=int, default=256)
    p.add_argument("--temp", type=float, default=1.0)
    p.add_argument("--allow-non-ascii", action="store_true", default=False)
    p.add_argument(
        "--run-gcg-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to run the standard GCG baseline (disabled by default).",
    )
    p.add_argument(
        "--run-ablation-baseline",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Whether to run the directional ablation baseline (disabled by default).",
    )
    p.add_argument("--disable-llamaguard", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--activation-obj",
        choices=["negative", "zero", "global_zero", "layer_zero_all", "token_all_layers"],
        default="layer_zero_all",
        help="Activation objective for activation-GCG. "
             "'layer_zero_all' (default) minimizes projection magnitude across all tokens at the chosen layer; "
             "'token_all_layers' minimizes projection magnitude at the chosen token across all layers; "
             "'zero' minimizes at a single (layer,pos); 'global_zero' minimizes across all layers/tokens; "
             "'negative' pushes projection negative at (layer,pos).",
    )
    p.add_argument(
        "--activation-score-mode",
        choices=["global", "local", "token_all_layers"],
        default="global",
        help="Scoring mode for activation-GCG candidate selection. "
             "'global' scores candidates over all layers/tokens; "
             "'local' uses only the canonical (layer,pos); "
             "'token_all_layers' aggregates across all layers at the chosen token.",
    )
    p.add_argument(
        "--conversation-template",
        default=None,
        help="FastChat conversation template name (defaults to llama-2).",
    )
    p.add_argument("--harmless-limit", type=int, default=100, help="Max harmless/harmful prompts to evaluate (use 0 for all)")
    p.add_argument("--evaluate-only", action="store_true", help="Skip attacks/generation and re-evaluate existing completions")
    return p.parse_args()


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)
    return path


DEFAULTS = {
    "llama2": {
        "direction_path": LLAMA2_DIRECTION_PATH,
        "direction_meta": LLAMA2_DIRECTION_META,
        "conversation_template": "llama-2",
        "model_cls": Llama2Model,
    }
}


def infer_model_family(model_path: str) -> str:
    # Currently only LLaMA-2 is supported.
    return "llama2"


def load_direction(direction_path, direction_meta):
    direction = torch.load(direction_path, map_location="cpu")
    if direction_meta and os.path.exists(direction_meta):
        with open(direction_meta) as f:
            meta = json.load(f)
        layer = meta["layer"]
        pos = meta["pos"]
    else:
        layer = 0
        pos = -1
    return direction, layer, pos


def build_params(model_path, device, conversation_template):
    # Minimal params object for get_workers
    params = SimpleNamespace()
    params.tokenizer_paths = [model_path]
    params.tokenizer_kwargs = [{"use_fast": False}]
    params.model_paths = [model_path]
    params.model_kwargs = [{"low_cpu_mem_usage": True, "use_cache": False}]
    params.conversation_templates = [conversation_template]
    params.devices = [device]
    params.num_train_models = 1
    return params


def sample_harmful_prompts(n_train):
    """Default harmful prompt sampling (used by activation-GCG): refusal_direction split."""
    harmful = load_dataset_split(harmtype="harmful", split="train", instructions_only=True)
    random.seed(42)
    random.shuffle(harmful)
    return harmful[:n_train]


def sample_advbench_pairs(n_train):
    """
    Sample paired (goal, target) from AdvBench harmful_behaviors.csv to mirror llm-attacks defaults.
    """
    import pandas as pd
    adv_path = os.path.join("third_party", "gcg", "data", "advbench", "harmful_behaviors.csv")
    if not os.path.exists(adv_path):
        return None, None
    df = pd.read_csv(adv_path)
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)
    goals = df["goal"].tolist()[:n_train]
    targets = df["target"].tolist()[:n_train]
    return goals, targets


def run_activation_gcg(args, direction, layer, pos, conversation_template):
    print("=== Running activation-based GCG ===")
    params = build_params(args.model_path, args.device, conversation_template)
    workers, test_workers = get_workers(params)

    goals = sample_harmful_prompts(args.n_train)
    targets = ["Sure."] * len(goals)  # placeholder; not used in activation objective

    managers = {"AP": ActAttackPrompt, "PM": ActPromptManager, "MPA": ActMultiPromptAttack}

    score_mode = (
        "token_all_layers"
        if args.activation_obj == "token_all_layers"
        else args.activation_score_mode
    )

    attack = ActMultiPromptAttack(
        goals,
        targets,
        workers,
        direction=direction,
        layer=layer,
        pos=pos,
        act_obj=args.activation_obj,
        score_mode=score_mode,
        control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        managers=managers,
    )
    suffix, loss = attack.run(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        topk=args.topk,
        temp=args.temp,
        allow_non_ascii=args.allow_non_ascii,
        target_weight=1.0,
        control_weight=0.0,
        test_steps=50,
        stop_on_success=False,
        verbose=False,
        filter_cand=True,
        anneal=False,
    )[:2]

    for w in workers + test_workers:
        w.stop()
    return suffix


def run_standard_gcg(args, conversation_template):
    print("=== Running standard GCG baseline ===")
    params = build_params(args.model_path, args.device, conversation_template)
    workers, test_workers = get_workers(params)
    goals, targets = sample_advbench_pairs(args.n_train)
    if goals is None or targets is None:
        # Fallback: use refusal_direction harmful prompts and build targets with instruction
        goals = sample_harmful_prompts(args.n_train)
        targets = [f"Sure, here is how to {g}" if g else "Sure, here is how to" for g in goals]
    managers = {"AP": GcgAttackPrompt, "PM": GcgPromptManager, "MPA": GcgMultiPromptAttack}
    attack = GcgMultiPromptAttack(
        goals,
        targets,
        workers,
        control_init="! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! ! !",
        managers=managers,
    )
    suffix, loss = attack.run(
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        topk=args.topk,
        temp=args.temp,
        allow_non_ascii=args.allow_non_ascii,
        target_weight=1.0,
        control_weight=0.1,
        test_steps=50,
        stop_on_success=False,
        verbose=False,
        filter_cand=True,
        anneal=False,
    )[:2]
    for w in workers + test_workers:
        w.stop()
    return suffix


def generate_completions(model_base, dataset, suffix=None, fwd_pre_hooks=None, fwd_hooks=None, max_new_tokens=128):
    fwd_pre_hooks = fwd_pre_hooks or []
    fwd_hooks = fwd_hooks or []
    # Append suffix to instruction if provided
    augmented = []
    for row in dataset:
        inst = row["instruction"] + (f" {suffix}" if suffix else "")
        augmented.append({"instruction": inst, "category": row.get("category", "unknown")})
    return model_base.generate_completions(
        augmented,
        fwd_pre_hooks=fwd_pre_hooks,
        fwd_hooks=fwd_hooks,
        batch_size=8,
        max_new_tokens=max_new_tokens,
    )


def load_eval_sets(harmless_limit: int = 0):
    harmful_eval = load_dataset_split(harmtype="harmful", split="test")
    harmless_eval = load_dataset_split(harmtype="harmless", split="test")
    # Apply the limit symmetrically so harmful and harmless use the same number of prompts
    if harmless_limit and harmless_limit > 0:
        harmful_eval = harmful_eval[:harmless_limit]
        harmless_eval = harmless_eval[:harmless_limit]
    return harmful_eval, harmless_eval


def main():
    # Ensure CUDA works with multiprocessing
    try:
        mp.set_start_method("spawn", force=True)
    except RuntimeError:
        # start method was already set in this process; that's fine
        pass

    args = parse_args()
    random.seed(args.seed)
    ensure_dir(args.output_dir)
    ensure_dir(os.path.join(args.output_dir, "completions"))

    model_family = infer_model_family(args.model_path)
    defaults = DEFAULTS[model_family]

    conversation_template = args.conversation_template or defaults["conversation_template"]

    direction, layer, pos = load_direction(args.direction_path, args.direction_meta)
    print(f"Loaded direction: layer={layer}, pos={pos}, norm={direction.norm().item():.4f}")

    # Prepare model for evaluation (reuse refusal-direction utilities)
    model_base_cls = defaults["model_cls"]
    model_base = model_base_cls(args.model_path)
    harmful_eval, harmless_eval = load_eval_sets(args.harmless_limit)

    variants = []

    if args.evaluate_only:
        print("=== Evaluate-only mode: skipping attacks and generation ===")
        # Use existing completions from disk; include a no-suffix baseline if present.
        variants = [("baseline", None, [], []), ("activation_gcg", None, [], [])]
        if args.run_gcg_baseline:
            variants.append(("gcg", None, [], []))
        if args.run_ablation_baseline:
            ablation_pre, ablation_hooks = get_all_direction_ablation_hooks(model_base, direction)
            variants.append(("ablation", None, ablation_pre, ablation_hooks))
    else:
        # Attack runs: first generate a no-suffix baseline, then run standard GCG, then activation-GCG.
        variants.append(("baseline", None, [], []))

        gcg_suffix = None
        if args.run_gcg_baseline:
            gcg_suffix = run_standard_gcg(args, conversation_template)
            print("Standard GCG suffix:", gcg_suffix)

        activation_suffix = run_activation_gcg(args, direction, layer, pos, conversation_template)
        print("Activation-GCG suffix:", activation_suffix)

        variants.append(("activation_gcg", activation_suffix, [], []))
        if gcg_suffix:
            variants.append(("gcg", gcg_suffix, [], []))

        if args.run_ablation_baseline:
            ablation_pre, ablation_hooks = get_all_direction_ablation_hooks(model_base, direction)
            variants.append(("ablation", None, ablation_pre, ablation_hooks))

    # Generate completions (if needed) and evaluate
    for name, suffix, pre_hooks, hooks in variants:
        harm_path = os.path.join(args.output_dir, "completions", f"{name}_harmful.json")
        harmless_path = os.path.join(args.output_dir, "completions", f"{name}_harmless.json")

        if not args.evaluate_only:
            print(f"=== Generating completions for {name} ===")
            harm_comp = generate_completions(
                model_base, harmful_eval, suffix=suffix, fwd_pre_hooks=pre_hooks, fwd_hooks=hooks, max_new_tokens=128
            )
            harmless_comp = generate_completions(
                model_base, harmless_eval, suffix=suffix, fwd_pre_hooks=pre_hooks, fwd_hooks=hooks, max_new_tokens=128
            )

            json.dump(harm_comp, open(harm_path, "w"), indent=4)
            json.dump(harmless_comp, open(harmless_path, "w"), indent=4)
        else:
            print(f"=== Evaluate-only: loading completions for {name} from disk ===")
            if not (os.path.exists(harm_path) and os.path.exists(harmless_path)):
                print(f"=== Evaluate-only: skipping {name}, missing completions in {args.output_dir} ===")
                continue
            with open(harm_path) as f:
                harm_comp = json.load(f)
            with open(harmless_path) as f:
                harmless_comp = json.load(f)

        # Jailbreak evals (substring + optional LlamaGuard2)
        meths = ["substring_matching"]
        if not args.disable_llamaguard and os.getenv("TOGETHER_API_KEY"):
            meths.append("llamaguard2")

        print(f"=== Evaluating jailbreak metrics for {name} ===")
        eval_harm = evaluate_jailbreak(
            completions=harm_comp,
            methodologies=meths,
            evaluation_path=os.path.join(args.output_dir, f"{name}_harmful_eval.json"),
        )
        eval_harmless = evaluate_jailbreak(
            completions=harmless_comp,
            methodologies=meths,
            evaluation_path=os.path.join(args.output_dir, f"{name}_harmless_eval.json"),
        )

        print(f"{name} substring ASR (harmful): {eval_harm.get('substring_matching_success_rate', 'n/a')}")

        # CE / perplexity on harmless completions (acts as on-distribution)
        print(f"=== Evaluating CE/perplexity for {name} on harmless completions ===")
        loss_metrics = evaluate_loss(
            model_base,
            fwd_pre_hooks=pre_hooks,
            fwd_hooks=hooks,
            batch_size=2,
            n_batches=256,
            dataset_labels=["alpaca_custom_completions"],
            completions_file_path=harmless_path,
        )
        json.dump(
            loss_metrics,
            open(os.path.join(args.output_dir, f"{name}_loss_eval.json"), "w"),
            indent=4,
        )

    print("All done. Artifacts in", args.output_dir)


if __name__ == "__main__":
    main()
