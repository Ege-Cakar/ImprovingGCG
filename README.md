# Improving GCG: Activation-Guided and Soft-GCG

This repository contains the code and experiments for a CS2881 project on improving the Greedy Coordinate Gradient (GCG) jailbreak attack on aligned language models such as LLaMA-2-7B-Chat and Gemma 3.

We explore two complementary directions:

- **Activation-Guided GCG** (in `Activation-Guided-GCG/`): a white-box variant of GCG that keeps the original greedy discrete suffix search, but replaces the log-likelihood objective with losses that directly suppress projections onto learned *refusal directions* in the model’s residual stream. These directions come from the “Refusal Direction” pipeline and represent a one-dimensional safety subspace in the model’s residual stream. Our activation-based objectives (single-layer, negative, layer-wide, token-wide, and global) explicitly target this subspace rather than only the output logits.

- **Soft-GCG** (in `Soft-GCG/`): a continuous relaxation of GCG that optimizes a “soft” suffix using Gumbel-Softmax over the vocabulary, then projects back to discrete tokens via argmax. It uses temperature annealing schedules (including a three-phase “slushy” schedule) and either cross-entropy or Carlini–Wagner-style losses to efficiently search the suffix space before snapping back to hard tokens.

At a high level, our results show that:

- Activation-based objectives can achieve higher attack success per optimization step than standard GCG, suggesting that mechanistic, representation-level targets can make discrete suffix optimization more sample-efficient.
- Soft-GCG can match GCG’s attack success rate while running orders of magnitude faster, and reveals a correlation between model capability and robustness in the Gemma 3 family (smaller models remain jailbreakable, larger models resist the attack).
- Across both lines of work, we evaluate on harmful and harmless prompt splits using substring matching, LlamaGuard2, HarmBench, and internal activation metrics (cosine similarity and Frobenius distance between suffix-induced and ablated activations).

For full details, motivation, and quantitative results, see `Final_Paper.pdf`.

## Using Activation-Guided-GCG

The `Activation-Guided-GCG/` directory contains the activation-based GCG implementation and all experiments on LLaMA-2-7B-Chat.

Basic workflow:

1. **Set up the environment**
   - From the repo root:
     - `cd Activation-Guided-GCG`
     - Install dependencies (e.g., `pip install -r requirements.txt` or use `pyproject.toml` with your preferred tool).
   - The `third_party/refusal_direction` and `third_party/gcg` directories are vendored; no extra cloning is required.

2. **Run activation-guided GCG and baselines**
   - From `Activation-Guided-GCG/`, run the unified pipeline:
     - `python activation_pipeline.py --model-path meta-llama/Llama-2-7b-chat-hf --direction-path third_party/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction.pt --direction-meta third_party/refusal_direction/pipeline/runs/llama-2-7b-chat-hf/direction_metadata.json --output-dir outputs/activation_gcg --activation-obj layer_zero_all --run-gcg-baseline --run-ablation-baseline`
   - This will:
     - Learn an activation-guided universal suffix,
     - Optionally run the standard GCG baseline and refusal-direction ablation,
     - Generate harmful/harmless completions and write evaluation JSONs under `outputs/activation_gcg/`.

3. **Evaluate and analyze**
   - To re-score existing completions with substring, LlamaGuard2, or HarmBench:
      - `python scripts/eval_safety.py --output-dir outputs/activation_gcg --variants baseline,activation_gcg,gcg,ablation --methods substring_matching,llamaguard2,harmbench`
   - To study how suffixes interact with the refusal direction across the whole network, use:
      - `final_analysis/generate_all_completions.py` and `final_analysis/compute_activation_frobenius.py` (see the docstrings in each script for example commands).

### Key scripts and arguments (Activation-Guided-GCG)

**`activation_pipeline.py`**

- `--model-path`: Hugging Face path or local path for the chat model (default `meta-llama/Llama-2-7b-chat-hf`).
- `--direction-path`: Path to the precomputed refusal direction `.pt` file (defaults to the bundled Llama-2 direction).
- `--direction-meta`: JSON with `{layer, pos}` metadata for the direction (defaults to bundled metadata).
- `--output-dir`: Root directory where completions and eval JSONs are saved (default `outputs/activation_gcg`).
- `--n-train`: Number of harmful train prompts to use when optimizing the suffix.
- `--n-steps`: Number of discrete optimization steps for activation-GCG (and GCG baseline when enabled).
- `--batch-size`, `--topk`, `--temp`: Standard GCG hyperparameters controlling candidate sampling and evaluation.
- `--allow-non-ascii`: Allow non-ASCII tokens in the suffix search space.
- `--run-gcg-baseline`: Enable optimization of a standard GCG suffix for comparison.
- `--run-ablation-baseline`: Enable a refusal-direction ablation baseline (hooks only, no suffix).
- `--disable-llamaguard`: When `False`, and with `TOGETHER_API_KEY` set, adds LlamaGuard2-based ASR evaluation.
- `--device`: Device string for model and workers (e.g., `cuda:0`).
- `--seed`: Random seed for sampling and attack reproducibility.
- `--activation-obj`: Activation objective to optimize (`negative`, `zero`, `global_zero`, `layer_zero_all`, `token_all_layers`).
- `--activation-score-mode`: How candidate suffixes are scored (`global`, `local`, `token_all_layers`).
- `--conversation-template`: FastChat conversation template name (defaults to a Llama-2 chat template).
- `--harmless-limit`: Max number of harmful/harmless test prompts to evaluate (0 = full splits).
- `--evaluate-only`: Skip attacks and generation, and only re-evaluate existing completions under `--output-dir`.

**`scripts/eval_safety.py`**

- `--output-dir`: Directory containing the `completions/` subdir created by `activation_pipeline.py`.
- `--variants`: Comma-separated list of variants to evaluate (e.g., `baseline,activation_gcg,gcg,ablation`).
- `--methods`: Comma-separated list of evaluation methodologies for `evaluate_jailbreak` (e.g., `substring_matching,llamaguard2,harmbench`).
- `--split`: Which completions to score (`harmful`, `harmless`, or `both`).

**`scripts/eval_substring_baseline.py`**

- `--model-path`: HF or local path to LLaMA-2-Chat (used directly via transformers).
- `--output-dir`: Directory where baseline completions and substring ASR results will be written.
- `--harmless-limit`: Max number of harmful/harmless prompts to use from the refusal-direction test splits (0 = all).

**`scripts/generate_suffix_completions.py`**

- `--model-path`: HF or local path to LLaMA-2-Chat, loaded via the refusal-direction `Llama2Model` wrapper.
- `--suffix`: The suffix string to append to every instruction (e.g., a suffix learned elsewhere).
- `--output-dir`: Root directory where `completions/<name>_harmful.json` and `<name>_harmless.json` will be saved.
- `--name`: Logical variant name used in filenames and downstream evaluation (e.g., `gcg_manual`).
- `--harmless-limit`: Max number of harmful/harmless prompts to generate completions for (0 = all).

**`final_analysis/generate_all_completions.py`**

- `--model-path`: HF or local path to LLaMA-2-Chat (via `Llama2Model`).
- `--direction-path`: Path to the refusal direction `.pt` file.
- `--direction-meta`: JSON with `{layer, pos}` metadata.
- `--suffix-csv`: CSV with columns `method,loss,suffix` (e.g., `final_analysis/method_suffix_summary.csv`).
- `--output-dir`: Directory where completions for baseline, ablation, and all listed methods will be generated.
- `--harmless-limit`: Optional limit on harmful/harmless test prompts (0 = full splits).

**`final_analysis/compute_activation_frobenius.py`**

- `--model-path`: HF or local path to LLaMA-2-Chat (via `Llama2Model`).
- `--direction-path`: Path to the refusal direction `.pt` file.
- `--direction-meta`: JSON with `{layer, pos}` metadata.
- `--suffix-csv`: CSV mapping method names to their optimized suffixes.
- `--output`: Path for the JSON file containing cosine/Frobenius statistics (default `final_analysis/activation_distances.json`).
- `--n-examples`: Number of harmful test prompts to sample for the activation-distance analysis.
- `--device`: Device where the model should be placed (e.g., `cuda:0`).

## Using Soft-GCG

The `Soft-GCG/` directory contains the continuous relaxation of GCG and the experiments on Gemma 3 and hybrid Soft+GCG schedules.

Basic workflow:

1. **Set up the environment**
   - From the repo root:
     - `cd Soft-GCG`
     - Install dependencies (e.g., using `pyproject.toml` and `uv.lock`, or your preferred tool).

2. **Run pure Soft-GCG on Gemma 3**
   - Use `run_sgcg_gemma.py` to optimize a universal soft suffix for a chosen Gemma 3 model:
     - `python run_sgcg_gemma.py --model gemma3:1b`
   - This:
     - Loads the chosen Gemma model,
     - Runs the Soft-GCG optimization (Gumbel-Softmax + temperature annealing),
     - Evaluates ASR using substring refusal detection,
     - Writes logs and CSVs under a timestamped `runs/` directory (including the final suffix and ASR).

3. **Run Soft vs GCG sweeps**
   - Use `sweep_script.py` to sweep over different ratios of Soft vs discrete GCG steps (e.g., pure GCG, pure Soft, 50/50, heavy Soft warmup):
     - `python sweep_script.py`
     - For a fast sanity check, use `--smoke-test` to run on a tiny subset of data.
   - Each sweep run writes `sweep_results.csv` with per-config ASR and suffixes into `runs/<timestamp>/`.

4. **Aggregate and plot sweep results**
   - Use `eval_sweep_bars.py` to re-evaluate each suffix over a prompt set and produce aggregated statistics and plots:
     - `python eval_sweep_bars.py --results_csv runs/RUN_.../sweep_results.csv --prompts_file advbench_prompts.txt --refusals_file refusal_substrings.txt --output_csv final_stats.csv`
   - This produces:
     - A CSV with mean ASR and SEM per configuration (`final_stats.csv`),
     - A bar chart (`final_bar_chart.png`) and an efficiency curve (`final_efficiency_curve.png`) showing the trade-off between Soft vs discrete steps.

### Key scripts and arguments (Soft-GCG)

**`run_sgcg_gemma.py`**

- `--model`: Model key specifying which Gemma 3 model to use (e.g., `gemma3:270m`, `gemma3:1b`, `gemma3:4b`, `gemma3:12b`, `gemma3:27b`); mapped internally to HF model names.
- `--smoke-test`: If set, runs a tiny, fast experiment (few prompts, very few steps) to verify the pipeline.
- `--2` / `--variant2`: Enable “Variant 2” Soft-GCG that uses the 3-phase temperature schedule and a Carlini–Wagner-style loss instead of pure cross-entropy.

**`sweep_script.py`**

- `--smoke-test`: If set, runs a small smoke-test sweep with a couple of configs and a truncated train/test set; without this flag, runs the full sweep over several Soft/GCG step allocations and writes `sweep_results.csv`.

**`eval_sweep_bars.py`**

- `--results_csv`: Path to a sweep results file (e.g., `runs/RUN_.../sweep_results.csv`) containing labels, Soft/GCG step counts, and suffixes.
- `--prompts_file`: Text file with one harmful prompt per line (e.g., an AdvBench-style subset) to evaluate ASR on.
- `--refusals_file`: Text file with one refusal substring per line (e.g., “I cannot”, “I’m unable”) used for substring-based refusal detection.
- `--output_csv`: Path where aggregated stats (mean ASR and SEM per configuration) will be written (default `final_stats.csv`).
- `--num_experiments`: Number of independent evaluation runs per configuration to estimate variability (used to compute SEM).
