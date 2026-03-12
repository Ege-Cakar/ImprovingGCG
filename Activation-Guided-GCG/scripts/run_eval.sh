#!/usr/bin/env bash
set -euo pipefail

if [ -z "${TOGETHER_API_KEY:-}" ]; then
  echo "Error: TOGETHER_API_KEY is not set" >&2
  exit 1
fi

# Activation-GCG reruns + baseline
for dir in rerun_zero rerun_negative rerun_global_zero rerun_layer_zero_all rerun_token_all_layers; do
  echo "=== Evaluating $dir ==="
  python scripts/eval_safety.py \
    --output-dir outputs/$dir \
    --variants activation_gcg,baseline \
    --methods llamaguard,harmbench \
    --split harmful
done

# GCG and ablation (original run)
echo "=== Evaluating gcg + ablation ==="
python scripts/eval_safety.py \
  --output-dir outputs/activation_gcg \
  --variants gcg,ablation \
  --methods llamaguard,harmbench \
  --split harmful
