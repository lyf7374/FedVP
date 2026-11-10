# Federated baseline protocols

The main paper compares FedVP with eight conventional federated baselines and two model-heterogeneous approaches. The result tables report the medical-QA reproductions used in the paper, not scores copied from the original publications.

## Included training wrapper

`scripts/run_federated_baselines.py` supplies the four generator-tuning implementations for FedIT, FFA-LoRA, FLoRA, and FedPrompt. Their complete settings are in `configs/baseline_training.json`. The shared recipe uses all five clients, two communication rounds, one local epoch per round, effective batch size 2, and client-training-size-weighted aggregation. It uses the same processed dataset files and final-answer parser as FedVP.

| Method | Trainable parameters | Local optimizer / LR | Rank / alpha | Aggregation |
| --- | --- | --- | --- | --- |
| FedIT | LoRA A and B on `q_proj` | AdamW / `3e-4` | 32 / 16 | Weighted A and B averages |
| FFA-LoRA | LoRA B on `q_proj`, `v_proj`; shared frozen A | SGD / `5e-3` | 32 / 16 | Weighted B average |
| FLoRA | Fresh A and B on `q_proj`, `v_proj` per client and round | AdamW / `3e-4` | 32 / 32 | Adapter stacking and merge |
| FedPrompt | 20 soft prompt tokens at the sequence head | Adam / `0.3` | -- | Weighted prompt-embedding average |

The FFA-LoRA and FedPrompt losses use assistant targets only; the FedIT and FLoRA losses include prompt and target tokens. LoRA dropout is 0.05. FedIT/FLoRA use linear learning-rate decay and gradient clipping at 1.0; FFA-LoRA/FedPrompt use constant learning rates. FedPrompt retains its virtual tokens during inference; its adapter is not merged into a generator checkpoint.

For example, after data and model preparation:

```bash
python scripts/run_federated_baselines.py train \
  --method fedit --backbone Meta-Llama-3-8B-Instruct \
  --model-path models/Meta-Llama-3-8B-Instruct \
  --config configs/baseline_training.json --output runs/fedit-llama

python scripts/run_federated_baselines.py predict \
  --run-dir runs/fedit-llama --model-path models/Meta-Llama-3-8B-Instruct
```

The prediction wrappers retain raw native-context-limited outputs in `pending_reviews` and leave them unresolved. Stop and review such a record before rerunning a four-method baseline prediction; the classical-FL wrapper additionally skips pending IDs while completing other questions. Ordinary budget truncations receive larger budgets. Historical experiment-specific resolution records are not distributed as a policy for new generations; resolve reviewed outputs before reporting a complete evaluation.

These conventional generator-tuning rows use one generated answer per question. FedCoT selects with its trained discriminator; it is not a Self-Consistency row.

## Classical federated optimization

`scripts/run_classic_fl_baselines.py` provides FedProx, FedAdam, and FedAdagrad with settings in `configs/classic_fl_baselines.json`. It uses the same `train`/`predict` interface and processed datasets as the four-method wrapper.

- **FedProx:** the FedIT LoRA/local-training recipe with proximal coefficient `mu=0.01`.
- **FedAdam:** the same local recipe, sample-weighted client deltas, server learning rate `1e-4`, `tau=1e-3`, and `(beta1, beta2)=(0.9, 0.99)`.
- **FedAdagrad:** the same local recipe, sample-weighted client deltas, server learning rate `1e-3`, and `tau=1e-3`.

For FedAdam/FedAdagrad, the second moment starts at `tau**2`; FedAdam's first moment starts at zero and bias correction is disabled. Their local optimizer and linear schedule match FedIT.

## FedCoT

The paper uses a Longformer-base-4096 discriminator; three rounds, one local epoch per round; client LoRA ranks `(4,32,32,16,4)` in A-E order; alpha equal to rank; dropout 0.1; query/value targets; learning rate `2e-4`; effective batch 16; cosine schedule with 10% warm-up and weight decay 0.01. The server stacks discriminator adapters and averages classification heads using retained-candidate counts. The final round is evaluated. Its result snapshot and protocol are included; the full FedCoT training pipeline requires the external author implementation and is not bundled here.

## Model-heterogeneous comparison

FedMD and FedMKT use the five generator architectures listed in [the protocol](protocol.md). The original training data are divided into 1,442 shared reference questions and 12,979 private questions. Each client receives its own private partition and the common labeled reference union. The main paper's FedVP row instead uses its original Heterogeneous pool trained on the full client-local partitions, without a shared-reference requirement.

Both distillation baselines initialize rank-16 LoRA independently, with alpha 32, dropout 0.05, AdamW at `1e-4`, effective batch 32, microbatch 1, and seed 42.

### FedMD

The medical-QA adaptation uses valid-answer-option logits. It starts with one public supervised epoch and one private supervised epoch. Each of two communication rounds averages the five clients' raw public option logits equally, performs one public epoch minimizing mean absolute error to that consensus, and follows with one private supervised epoch. The supervised target retains the available explanation and canonical answer.

### FedMKT

An independently initialized `Qwen/Qwen2.5-7B-Instruct` server sees only the public reference union. Each of five rounds performs one epoch of private supervised client training, one epoch of public-data distillation to the server, and one epoch of public-data distillation back to each client. It exchanges top-128 token logits, uses MinED alignment and DualMinCE teacher selection, temperature 1, and loss `0.9 * LM + 0.1 * distillation`.

The reported targets are the final LLaMA-3-8B and Qwen2.5-7B **client** models. The FedMKT server is a distinct Qwen checkpoint and is not the reported Qwen target.

For both methods, the without-SC result selects candidate 0 from the saved eight-response pool. The SC result votes over those same eight responses, with valid-answer vote ties resolved by first occurrence. These baseline implementations are LLM adaptations of the source methods; their full distributed pipelines are not bundled in this release.

## Source implementations and papers

| Method | Source |
| --- | --- |
| FedIT | [FederatedGPT-Shepherd](https://github.com/JayZhang42/FederatedGPT-Shepherd) |
| FFA-LoRA | [Paper](https://arxiv.org/abs/2403.12313); [FederatedScope LLM baseline configuration](https://github.com/alibaba/FederatedScope/blob/dev/llm/federatedscope/llm/baseline/exp_yaml/gsm/gsm_federate.yaml) |
| FLoRA | [FederatedLLM](https://github.com/ziyaow1010/FederatedLLM) |
| FedPrompt | [Paper](https://arxiv.org/abs/2208.12268) |
| FedProx / FedOpt LLM adaptation | [OpenFedLLM](https://github.com/rui-ye/OpenFedLLM); [Adaptive Federated Optimization](https://arxiv.org/abs/2003.00295) |
| FedCoT | [Author code](https://github.com/DIaacKr/FedCoT); [paper v2](https://arxiv.org/abs/2508.10020v2) |
| FedMD | [FedMD_clean](https://github.com/diogenes0319/FedMD_clean); [paper](https://arxiv.org/abs/1910.03581) |
| FedMKT | [FATE-LLM](https://github.com/FederatedAI/FATE-LLM) |

Upstream projects and model snapshots retain their own licenses. They are obtained separately; the links above do not imply that their source trees or pretrained weights are distributed with this repository.
