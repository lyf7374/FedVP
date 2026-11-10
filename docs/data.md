# Data preparation

The five public QA datasets are treated as separate clients. All split sizes and preprocessing follow the fixed protocol used for the paper.

| Client | Dataset | Train | Validation | Test |
| --- | --- | ---: | ---: | ---: |
| A | BioASQ yes/no | 494 | 0 | 124 |
| B | MedMCQA | 2,000 | -- | 4,183 |
| C | MedQA | 10,178 | 1,272 | 1,273 |
| D | MMLU-Med | 1,299 | 162 | 163 |
| E | PubMedQA | 450 | 50 | 500 |
| Total | | 14,421 | | 6,243 |

## Sources and splits

- [BioASQ](https://participants-area.bioasq.org/datasets/): yes/no questions from Task 7B-11B GoldenEnriched files. Read years 11 through 7 and batches in numerical order, shuffle using Python `random.Random(42)`, and split 80/20.
- [MedMCQA](https://huggingface.co/datasets/openlifescienceai/medmcqa): shuffle official training data with NumPy's seed-42 generator, then keep the first 2,000 records with non-null explanations. The official validation questions form the 4,183-question test set. The exported `validation.json` contains the same question source and is not an additional held-out validation set; it is shown as `--` in the paper.
- [MedQA](https://huggingface.co/datasets/bigbio/med_qa): retain the original English four-option split membership (`med_qa_en_4options_bigbio_qa`), with deterministic within-split shuffling.
- [MMLU](https://people.eecs.berkeley.edu/~hendrycks/data.tar): combine clinical knowledge, college biology, college medicine, high-school biology, medical genetics, professional medicine, and virology. For each subject, load test, validation, and dev files in that order; then use a seed-42 Python shuffle and an 80/10/10 split.
- [PubMedQA](https://huggingface.co/datasets/qiaojin/PubMedQA): use the 1,000 expert-labeled PQA-L records, shuffle with NumPy seed 42, then split into 450/50/500.

The loader retains the FedCoT processing functions and their option randomization. A single seed-42 option-shuffle generator is carried through train, validation, and test. Reordering source files or resetting that generator between splits changes the exported option labels.

## Local layout

`prepare_data.py` reads the FedCoT dataset processors from `upstream/FedCoT/src/dataset_loader.py`. Use the upstream acquisition instructions in the repository README first. Raw dataset files are downloaded on demand except BioASQ, whose GoldenEnriched files should be placed as follows:

```text
data/raw/bioasq/
  Task11BGoldenEnriched/11B1_golden.json ...
  Task10BGoldenEnriched/10B1_golden.json ...
  Task9BGoldenEnriched/9B1_golden.json ...
  Task8BGoldenEnriched/8B1_golden.json ...
  Task7BGoldenEnriched/7B1_golden.json ...
```

Run from the repository root:

```bash
python scripts/prepare_data.py --datasets pubmedqa medmcqa medqa mmlu bioasq --seed 42
```

Each dataset is exported to `data/processed/<dataset>/{train,validation,test}.json`, accompanied by `manifest.json`. A processed row contains `question`, `answer`, `answer_string`, `answer_letter`, `answer_idx`, `options`, and `options_str`.

The repository does not bundle raw questions, model weights, or generated response pools. Download source datasets and pretrained models under their respective terms.

## Reference subset for FedMD, FedMKT and public distillation

The fixed reference subset contains 49 BioASQ, 200 MedMCQA, 1,018 MedQA, 130 MMLU-Med, and 45 PubMedQA training questions (1,442 total). Selection uses NumPy `RandomState(42)`, sampling without replacement once per center in A-E order, with `round(0.1 * train_count)` records from each center. Preserve the selected original training indices.

The exact saved indices are included in [public_reference_ids.json](../configs/public_reference_ids.json). Each `original_train_index` is a zero-based index into the corresponding prepared `train.json`. The records are already ordered by center and original training index. Read them directly rather than drawing a new reference subset:

```python
import json
from pathlib import Path

spec = json.loads(Path("configs/public_reference_ids.json").read_text())
training = {
    center: json.loads(Path(f"data/processed/{center}/train.json").read_text())
    for center in spec["center_order"]
}
public = [
    training[item["dataset"]][item["original_train_index"]]
    for item in spec["records"]
]
assert len(public) == 1442
```

The CPU helper materializes those same rows, with source identities, to `data/processed/public/train.json`:

```bash
python scripts/prepare_public_reference.py materialize
```

To create one public candidate file for a generator, use `prepare_public_reference.py assemble` with its model ID and one `--pool DATASET=FILE` argument per original dataset. Each input can be a complete original training pool or a saved pool containing only the reference questions. The exporter preserves original responses and maps only the selected questions to public-union indices `0..1441`.

For a generator/dataset pair with no existing candidate pool, `generate_public_candidates.py` generates only its saved reference indices. For example:

```bash
python scripts/generate_public_candidates.py \
  --model-path models/Qwen2.5-7B-Instruct \
  --model-name Qwen2.5-7B-Instruct \
  --generator-id Qwen/Qwen2.5-7B-Instruct \
  --dataset bioasq \
  --output-dir candidates/public_subsets/Qwen2.5-7B-Instruct/bioasq
```

The resulting `candidates.json` retains original BioASQ training indices and can be passed to `assemble` as the BioASQ input. This entry point retains the original dataset-specific prompts and temperature/top-p schedule indexed by original 128-question blocks. Reuse existing candidate files wherever available. The five generator files assembled for public distillation all share the same 1,442 public question identities.

For FedMD/FedMKT, this leaves 12,979 private questions; each client receives its own private partition and the common reference union. FedVP's main Heterogeneous and Uniform results use the original full local partitions and do not reserve this subset. The public-distillation student uses the subset, while its original Heterogeneous teacher keeps the main experiment's training protocol.

## Candidate and score identity

A candidate pool is a JSON list of questions. Each question retains its original `sample_idx`, answer/options metadata, and eight candidates with stable `candidate_id` values. Preserve the raw response, parsed answer, correctness label and completion metadata. Score files identify both the question and candidate; aggregate only matching identities in the same original order.

The shared answer parser is `final-answer-v3-2026-10-21`. It extracts explicit final answers without consulting the reference answer. Ordinary length truncation triggers an increased generation budget. Native-context-limited outputs are retained for explicit review rather than silently converted to incorrect answers.
