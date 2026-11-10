# Reproducing training and selection

Run commands from the repository root. The scripts operate on explicit local model snapshots, candidate files and checkpoints. They do not submit cluster jobs or choose GPUs for you.

## Environments

Use Python 3.11. The recorded native-head verifier stack uses PyTorch 2.6.0, Transformers 4.52.3 and PEFT 0.15.2 (`requirements-verifier.txt`). The Skywork-Llama member also used Transformers 4.51.1 in the original run. SGLang generation used 0.4.6.post1 with the separate 4.51.1 stack. The complete CUDA 11.8 generation requirements are in `requirements-generation-cu118.txt`; install them in a separate Linux environment. `--server-python` can point to that environment's Python, or defaults to the invoking Python.

The public result reader and CPU selector require only standard-library Python. Data preparation requires NumPy and PyArrow. Choose the PyTorch CUDA wheel for your hardware and driver; model snapshots are downloaded separately using `scripts/download_models.py`.

## Candidate generation

Prepare all five datasets first. The original candidate sampler consumes its random schedule in a fixed dataset/split order even when generating a selected subset. This is why the example generating only BioASQ still reads the split sizes of the other datasets.

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_heterogeneous_candidates.py \
  --model-name Meta-Llama-3-8B-Instruct \
  --model-path models/Meta-Llama-3-8B-Instruct \
  --datasets bioasq,medmcqa,medqa,mmlu,pubmedqa --splits test
```

For each local training run, generate `--splits train` using that client's assigned generator and dataset. Output paths are:

```text
outputs/best_of_n/<generator>/<dataset>_best_of_n/<split>/<split>_best_of_n.json
```

The sampler produces eight candidates per question. It preserves the FedCoT prompt, native generator chat template, seed 42, and the recorded 128-question block temperature/top-p schedule. Ordinary length-limited responses receive a larger budget. Native-context exhaustion retains the raw attempts and stops for review; it is not automatically scored as incorrect. Repeated invocation preserves committed candidates and captured completed responses. The five exclusions shipped with this release apply only to their exact original saved identities, not arbitrary new responses.

Existing `--batch-start`/`--batch-end` options address half-open ranges of original 1,024-candidate blocks. Use distinct shards and the same original pool paths. `--merge-only` assembles finished shards without loading a model. Do not run two writers against the same shard.

## Local verifier training

Use the README example for each Heterogeneous client with the model IDs in `configs/paper_experiments.json`. Supply exactly one generator source per local verifier. The ArmoRM member requires `--verifier-kind armorm` to retain its regression/gating architecture. Default hyperparameters match the paper.

For Uniform, use Skywork-Llama v0.2 at all five centers, but run two separate sets of checkpoints:

- `runs/uniform_llama/<center>` using only LLaMA-3-8B candidates.
- `runs/uniform_qwen/<center>` using only Qwen2.5-7B candidates.

The single-worker trainer accumulates microbatches to effective batch 32 and saves optimizer, data cursor and RNG state. Resume the matching run with `--resume <output-dir>/checkpoint`. Do not point a new center or source run at an unrelated checkpoint.

## Score a trained member

Every member scores the **same** candidate file for a target/dataset combination. Example: score LLaMA-generated BioASQ answers with Client A's trained GRM:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/score_heterogeneous_verifier.py \
  --center bioasq --generator Meta-Llama-3-8B-Instruct \
  --candidates outputs/best_of_n/Meta-Llama-3-8B-Instruct/bioasq_best_of_n/test/test_best_of_n.json \
  --checkpoint runs/heterogeneous/client_A/checkpoint \
  --model-path models/GRM_Llama3.1_8B_rewardmodel-ft \
  --output-dir runs/scores/llama/bioasq/A
```

Repeat for B-E, changing checkpoint, model path and output directory. Scores are written to `candidate_scores.jsonl`. The candidate key is `(sample_idx, candidate_id)`; scoring also records center, source identity, logit and sigmoid score. Question shards use `--question-start`/`--question-end` offsets and separate output directories. Train-split teacher scoring uses `--split train`.

The pool is aggregated on CPU:

```bash
python scripts/evaluate_pool.py \
  --candidates outputs/best_of_n/Meta-Llama-3-8B-Instruct/bioasq_best_of_n/test/test_best_of_n.json \
  --scores A=runs/scores/llama/bioasq/A/candidate_scores.jsonl \
  --scores B=runs/scores/llama/bioasq/B/candidate_scores.jsonl \
  --scores C=runs/scores/llama/bioasq/C/candidate_scores.jsonl \
  --scores D=runs/scores/llama/bioasq/D/candidate_scores.jsonl \
  --scores E=runs/scores/llama/bioasq/E/candidate_scores.jsonl \
  --expected-questions 124 --output runs/evaluation/llama/bioasq.json
```

For score shards, repeat `--scores` with the same client name and a different JSONL path. Duplicate candidate keys are rejected. The selector uses the original stable scalar sigmoid on saved FP32 logits, averages probabilities, and keeps the first candidate on an exact tie, including sigmoid-saturation ties.

## Existing paper analyses

Reuse the saved scores with the following options; no additional model calls are needed:

| Analysis | Selector option |
| --- | --- |
| Candidate-budget figure | `--n 1`, `2`, `4`, or `8` |
| Leave one verifier out | `--drop-member A` (or B-E) |
| Mean-logit aggregation | `--mode mean_logit` |
| Maximum-probability aggregation | `--mode max_probability` |
| Member voting | `--mode member_vote` |
| Self-Consistency | `--mode sc` (omit score files) |
| Oracle | `--mode oracle` (omit score files) |

Member voting reproduces the reported alternative: each verifier votes for its highest-logit candidate; vote-count ties are resolved by summed member probabilities, then source order. This is distinct from Self-Consistency, which votes over extracted answers without verifiers.

For pretrained evaluation, `scripts/score_pretrained_verifier.py` accepts the same candidate/center/generator arguments plus `--model-path`, `--model-id` and `--verifier-kind`, without `--checkpoint`. Score one released Skywork-Llama verifier for the single-verifier row, or all five released heterogeneous verifiers for the pretrained-pool row. Aggregate using the default sigmoid rule.

## Five-dataset average

After completing all five full tests:

```bash
python scripts/summarize_accuracy.py \
  --result bioasq=runs/evaluation/llama/bioasq.json \
  --result medmcqa=runs/evaluation/llama/medmcqa.json \
  --result medqa=runs/evaluation/llama/medqa.json \
  --result mmlu=runs/evaluation/llama/mmlu.json \
  --result pubmedqa=runs/evaluation/llama/pubmedqa.json \
  --output runs/evaluation/llama/summary.json
```

The summary verifies the full denominators 124/4,183/1,273/163/500 and calculates the unweighted mean of their accuracies before rounding. Repeat for Qwen. For Table 4, use the same Heterogeneous checkpoints with each of the five target backbones.
