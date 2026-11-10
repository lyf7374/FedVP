# Experimental protocol

FedVP trains a verifier independently at each client and combines the resulting judgments at inference. Generator parameters are frozen. Verifier parameters are never averaged or merged.

The machine-readable specification is [paper_experiments.json](../configs/paper_experiments.json). Reported values are in [the paper result files](../results/).

## Heterogeneous clients

| Client | Dataset | Frozen generator | Verifier initialization |
| --- | --- | --- | --- |
| A | BioASQ | `meta-llama/Llama-3.2-3B-Instruct` | `Ray2333/GRM_Llama3.1_8B_rewardmodel-ft` |
| B | MedMCQA | `Qwen/Qwen2.5-7B-Instruct` | `Skywork/Skywork-Reward-Llama-3.1-8B-v0.2` |
| C | MedQA | `meta-llama/Meta-Llama-3-8B-Instruct` | `NCSOFT/Llama-3-OffsetBias-RM-8B` |
| D | MMLU-Med | `Qwen/Qwen2.5-3B-Instruct` | `RLHFlow/ArmoRM-Llama3-8B-v0.1` |
| E | PubMedQA | `Qwen/Qwen2.5-14B-Instruct` | `Skywork/Skywork-Reward-V2-Qwen3-8B` |

Each client uses only its assigned generator to produce eight responses per local training question. The same five-member Heterogeneous pool is reused for all five target generators. In the two-target comparison tables, only the target generator changes between the LLaMA and Qwen panels; the Heterogeneous verifier weights are identical.

## Uniform setting

Uniform consists of two independent runs. Every client uses LLaMA-3-8B-generated responses in one run, and Qwen2.5-7B-generated responses in the other. All verifiers start from Skywork-Llama v0.2. Each run trains five separate client verifiers, one per dataset. A Uniform pool is evaluated only with its matching target generator. Training does not mix the two response sources.

## Local training

Candidates receive binary correctness labels from the common final-answer parser and reference answers. All labeled training candidates are retained, including negative labels. Each verifier starts from its released pretrained backbone and native scoring components.

| Setting | Value |
| --- | --- |
| Epochs | 2 |
| Optimizer | AdamW, learning rate `1e-4`, betas `(0.9, 0.999)`, epsilon `1e-8` |
| Effective batch / microbatch | 32 / 1 |
| Schedule / weight decay / gradient clipping | Constant / 0 / none |
| LoRA rank / alpha / dropout | 16 / 32 / 0.05 |
| LoRA targets | `q_proj`, `k_proj`, `v_proj`, `o_proj`, `gate_proj`, `up_proj`, `down_proj` |
| Trainable weights | LoRA adapters and native scoring components |
| Precision | Frozen backbone BF16; trainable weights FP32 |
| Input | Native chat template, question plus complete response, without truncation |
| Seed | 42 |

For ArmoRM, the regression and gating layers are trainable and its released reward transformation remains fixed. The shared scoring interface returns the native scalar score; it does not replace native heads with a new common head.

The portable training entry point is `scripts/train_heterogeneous_verifier.py`. It also trains a Uniform member when supplied with a Skywork-Llama model and a single-source candidate file. Use `--help` for its file arguments.

## Inference and evaluation

For every candidate, each retained pool member produces a native score `z`. The default pool score is the arithmetic mean of `sigmoid(z)` across members. Select the candidate with the largest pooled score; exact ties select the first eligible candidate in saved order. A single pretrained or distilled verifier also ranks by its sigmoid score. The full question denominator is preserved when explicitly reviewed failed generation attempts are excluded from selection.

CoT selects the first saved candidate. Self-Consistency has no verifier: it votes over valid extracted answers, breaking vote ties by first occurrence. Accuracy is reported for each complete test set; Avg. is the unweighted mean of the five dataset accuracies, not accuracy pooled over all test questions.

The paper's scaling plot uses the same saved prefixes at `N=1,2,4,8`. The aggregation comparison uses mean probability, mean logit, maximum probability, and member majority voting. In the member-vote alternative, each member votes for its highest-raw-logit candidate; vote ties use summed member probabilities, then candidate order. The leave-one-out analysis removes one Heterogeneous member at a time and averages the remaining four probabilities.

## Optional distillation

Both variants use the original Heterogeneous pool from the main experiments as the teacher. Each teacher score is the mean of its five sigmoid scores. Students are initialized from `Skywork/Skywork-Reward-V2-Qwen3-8B` and train for two epochs with soft-target binary cross-entropy, the same LoRA recipe, effective batch 32, and learning rate `1e-4`. The teacher and generators remain frozen.

- **Public distillation:** one shared student, 1,442 reference questions, all five generator sources, eight responses per source and question: 57,680 training examples.
- **Local distillation:** five independent students, each using its center's original local questions and eight responses from its assigned generator: 115,368 examples in total.

For the paper's LLaMA-3-8B and Qwen2.5-7B panels, the public student is reused on all five tests. Each local student is used on its own center's test set under both target backbones. The teacher row is the same original Heterogeneous pool in both distillation comparisons.

`scripts/distill_verifier.py` accepts an ordered sources manifest. Each entry specifies `generator`, `candidates`, and `teacher_labels`; paths are relative to the manifest. Teacher-label JSONL records carry `sample_idx`, `candidate_id`, and `teacher_probability`. Public distillation uses five source entries and each local student uses one. The scoring entry point for a trained student is the same as for a trained native-head verifier.

The 1,442 questions are the same reference subset used for the FedMD/FedMKT comparison. They were not removed from the original teacher's client-local training sets. Public distillation is an additional training stage; the main FedVP method does not require this shared dataset.
