"""Shared candidate, batching, and input helpers from the original experiment implementation."""
import json
import os
from pathlib import Path
import random
ROOT = Path(__file__).resolve().parents[1]
MODEL_ID = "Skywork/Skywork-Reward-Llama-3.1-8B-v0.2"

def training_defaults():
    return dict(
        model_id=MODEL_ID, epochs=2, learning_rate=1e-4, effective_batch_size=32,
        microbatch_size=1, rank=16, alpha=32, dropout=0.05,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        train_score_head=True, optimizer="AdamW", optimizer_betas=[0.9, 0.999],
        optimizer_eps=1e-8, weight_decay=0.0, scheduler="constant", seed=42,
        backbone_dtype="bfloat16", trainable_dtype="float32", compute_dtype="bfloat16",
        attn_implementation="sdpa", gradient_checkpointing=True, checkpoint_steps=50,
        loss="hard_correctness_bce", candidate_filter="none", split="train",
        input_format="official_chat_user_question_assistant_full_response",
        truncation=False, input_limit="model_native_context", gradient_clipping=None,
    )


def configure_environment():
    cache = ROOT / "cache"
    paths = {"HF_HOME": cache / "huggingface", "HUGGINGFACE_HUB_CACHE": cache / "huggingface" / "hub",
             "HF_DATASETS_CACHE": cache / "huggingface" / "datasets", "XDG_CACHE_HOME": cache,
             "TORCH_HOME": cache / "torch", "TRITON_CACHE_DIR": cache / "triton",
             "TMPDIR": cache / "tmp", "TMP": cache / "tmp", "TEMP": cache / "tmp"}
    for key, path in paths.items():
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".writing")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                         encoding="utf-8")
    temporary.replace(path)


def load_candidate_sources(paths, center, split, *, allow_generation_failures=False, generator_name=None):
    """Flatten complete source pools without filtering or mixing client centers."""
    rows, sources = [], []
    resolved = [Path(path).resolve() for path in paths]
    if len(set(resolved)) != len(resolved):
        raise ValueError("Candidate source paths must be distinct")
    for source_index, path in enumerate(resolved):
        samples = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(samples, list) or not samples:
            raise ValueError(f"Expected a nonempty FedCoT question list: {path}")
        candidate_count = 0
        seen_samples = set()
        for sample in samples:
            if sample.get("dataset_name") != center:
                raise ValueError(f"dataset_name must be {center}: {path}")
            if sample.get("mode") != split:
                raise ValueError(f"mode must be {split}: {path}")
            sample_idx = sample["sample_idx"]
            if sample_idx in seen_samples:
                raise ValueError(f"Duplicate sample_idx {sample_idx} in {path}")
            seen_samples.add(sample_idx)
            question, candidates = sample["question"], sample["candidates"]
            if not isinstance(question, str) or not isinstance(candidates, list) or not candidates:
                raise ValueError(f"Question text and nonempty candidates required: {path}, {sample_idx}")
            seen_candidates = set()
            for candidate in candidates:
                from generation_failure_policy import candidate_failure_metadata
                failure = candidate_failure_metadata(candidate,
                    generator_name=(generator_name or sample.get("source_generator_id", "").rsplit("/", 1)[-1] or path.parent.parent.parent.name), dataset=center,
                    split=split, sample_idx=sample_idx, allow=allow_generation_failures)
                candidate_id = candidate["candidate_id"]
                if candidate_id in seen_candidates:
                    raise ValueError(f"Duplicate candidate_id {candidate_id}: {path}, {sample_idx}")
                seen_candidates.add(candidate_id)
                target = candidate["is_correct"]
                if not isinstance(target, (bool, int, float)) or target not in (0, 1):
                    raise ValueError(f"is_correct must be a hard 0/1 label: {path}, {sample_idx}")
                response = candidate["response_text"]
                if not isinstance(response, str):
                    raise ValueError("response_text must be a string")
                rows.append(dict(center=center, source_index=source_index, source_path=str(path),
                                 sample_idx=sample_idx, candidate_id=candidate_id, question=question,
                                 response_text=response, target=float(target),
                                 extracted_answer=candidate.get("extracted_answer"), **failure))
                candidate_count += 1
        sources.append(dict(path=str(path), source_index=source_index, center=center, split=split,
                            question_count=len(samples), candidate_count=candidate_count))
    if not rows:
        raise ValueError("At least one candidate source is required")
    return rows, sources


def encode_candidate(tokenizer, row, native_context):
    messages = [{"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["response_text"]}]
    ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False,
                                        truncation=False)
    if len(ids) > native_context:
        identity = (row.get("source_index"), row.get("sample_idx"), row.get("candidate_id"))
        raise ValueError(f"Candidate {identity} has {len(ids)} tokens, exceeding native context {native_context}")
    return ids


def collate_candidates(tokenizer, rows, native_context, device):
    import torch

    encoded = [encode_candidate(tokenizer, row, native_context) for row in rows]
    length = max(map(len, encoded))
    if tokenizer.pad_token_id is None:
        raise ValueError("The Skywork tokenizer must retain its published padding token")
    ids = torch.full((len(rows), length), tokenizer.pad_token_id, dtype=torch.long, device=device)
    mask = torch.zeros_like(ids)
    for index, sequence in enumerate(encoded):
        ids[index, :len(sequence)] = torch.tensor(sequence, dtype=torch.long, device=device)
        mask[index, :len(sequence)] = 1
    targets = torch.tensor([row["target"] for row in rows], dtype=torch.float32, device=device)
    return dict(input_ids=ids, attention_mask=mask, targets=targets)


def epoch_batches(count, effective_batch_size, seed, epoch):
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    for start in range(0, count, effective_batch_size):
        yield order[start:start + effective_batch_size]
