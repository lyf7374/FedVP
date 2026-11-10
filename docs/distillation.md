# Distilling the verifier pool

The teacher is the **original trained Heterogeneous pool**. Both public and local students start from `Skywork/Skywork-Reward-V2-Qwen3-8B`; neither starts from a teacher's fine-tuned weights. Students optimize soft-target BCE on the teacher's mean sigmoid score, with two epochs and the same LoRA/optimizer configuration as local verifier training. Gold correctness labels are not supplied to the student loss.

## Inputs

- **Public:** the 1,442 fixed reference questions in `configs/public_reference_ids.json`, each with eight saved responses from each of the five generators. Train one student on the five ordered sources (57,680 responses).
- **Local:** each center's original local training questions and eight responses from its assigned generator. Train five independent students, each on one source.

These are the fixed questions and candidate formats described in [data](data.md). Reuse existing source responses wherever available. Public candidate files retain `dataset_name="public"`, `mode="train"`, and public `sample_idx` order 0-1441. Their `uid` records the original center and training index. Local candidate files retain their original center and indices.

For the saved public IDs and the exact commands to materialize questions, assemble existing source responses, or generate missing public subsets, see [the reference-subset instructions](data.md#reference-subset-for-fedmd-fedmkt-and-public-distillation). These helpers do not draw a new reference subset.

## Teacher probabilities

For each source candidate file, run the trained-verifier scoring command for A-E using `--split train`. Set `--center public` for public inputs or the dataset name for local inputs. Use the original five Heterogeneous checkpoints for both variants. Then export soft labels:

```bash
python scripts/evaluate_pool.py \
  --candidates inputs/public_llama3b.json \
  --scores A=runs/teacher/public_llama3b/A/candidate_scores.jsonl \
  --scores B=runs/teacher/public_llama3b/B/candidate_scores.jsonl \
  --scores C=runs/teacher/public_llama3b/C/candidate_scores.jsonl \
  --scores D=runs/teacher/public_llama3b/D/candidate_scores.jsonl \
  --scores E=runs/teacher/public_llama3b/E/candidate_scores.jsonl \
  --expected-questions 1442 --teacher-labels \
  --output inputs/public_llama3b_teacher.jsonl
```

Repeat for the five public generator sources. For a local source, use its dataset's training count instead of 1,442. A teacher label row contains `sample_idx`, `candidate_id` and `teacher_probability`; it aligns with the source question/candidate order.

## Train the student

Write an ordered JSON sources manifest. This local example has one source; the public manifest has five entries in Client A-E generator order:

```json
[
  {
    "generator": "Llama-3.2-3B-Instruct",
    "candidates": "bioasq_train.json",
    "teacher_labels": "bioasq_teacher.jsonl"
  }
]
```

Paths resolve relative to the manifest. With the native student snapshot downloaded:

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/distill_verifier.py \
  --center bioasq --sources inputs/local_bioasq_sources.json \
  --model-path models/Skywork-Reward-V2-Qwen3-8B \
  --output-dir runs/local_student/bioasq
```

Use `--center public --sources inputs/public_sources.json --output-dir runs/public_student` for the shared student. The fixed defaults preserve two epochs, LR `1e-4`, global batch 32, microbatch 1, rank 16/alpha 32/dropout 0.05, complete inputs and seed 42. Resume only from that student's saved checkpoint with `--resume`.

## Evaluate

Use `scripts/score_heterogeneous_verifier.py` with the student checkpoint. The public student serves every center. Each local student serves only its own center's test set, using both LLaMA-3-8B and Qwen2.5-7B candidate pools. No test examples enter distillation. Apply the same single-verifier sigmoid selection and five-dataset average as in [reproduction](reproduction.md).

The reported teacher row is the original Heterogeneous pool. Optional student training is an additional compression stage; it is not required for FedVP's base collaboration protocol.
