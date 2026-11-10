"""Score one verifier x generator x candidate dataset, with resumable question shards.

--question-start/--question-end are zero-based half-open offsets in source question
order (not sample_idx values). Scores preserve actual sample_idx/candidate_id.
Use a distinct --output-dir per shard. Completed candidate scores are reused.
"""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time

from train_skywork_verifier import configure_environment, load_candidate_sources, collate_candidates, write_json
from heterogeneous_verifier import (load_verifier, forward_logits, question_range, row_key,
                                     score_record, read_saved_scores, select_predictions)


def run_scoring(args):
    configure_environment()
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    rows, sources = load_candidate_sources([args.candidates], args.center, args.split,
                                           allow_generation_failures=True, generator_name=args.generator.rsplit("/", 1)[-1])
    rows = question_range(rows, args.question_start, args.question_end)
    checkpoint = Path(args.checkpoint).resolve()
    manifest = json.loads((checkpoint / "manifest.json").read_text(encoding="utf-8"))
    training = manifest["config"]
    model_path = str(args.model_path.resolve()) if args.model_path else training["model_path"]
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    config = dict(center=args.center, generator=args.generator, split=args.split, sources=sources,
                  checkpoint=str(checkpoint), member_center=training["center"],
                  verifier_model_id=training["model_id"], verifier_kind=training["verifier_kind"],
                  model_path=model_path, question_start=args.question_start, question_end=args.question_end,
                  candidate_count=len(rows), batch_size=args.batch_size, truncation=False,
                  package_versions=dict(torch=torch.__version__, transformers=transformers.__version__,
                                        peft=peft.__version__))
    config_path = output / "scoring_config.json"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        for key in ("center", "generator", "sources", "checkpoint", "member_center", "verifier_model_id",
                    "verifier_kind", "question_start", "question_end", "candidate_count"):
            if previous[key] != config[key]:
                raise ValueError(f"Existing score shard belongs to different inputs: {key}")
    write_json(config_path, config)
    score_path = output / "candidate_scores.jsonl"
    saved = read_saved_scores(score_path, rows)
    missing = [row for row in rows if row_key(row) not in saved]
    started = time.monotonic()
    excluded = [row for row in missing if row.get("selection_eligible", True) is False]
    if excluded:
        # These attempts remain in N=8 coverage, but are never tokenized or scored.
        with score_path.open("a", encoding="utf-8") as stream:
            for row in excluded:
                record = score_record(row, None)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                saved[row_key(row)] = record
            stream.flush()
        missing = [row for row in missing if row.get("selection_eligible", True)]
    if missing:
        device = torch.device(args.device)
        model = load_verifier(model_path, training, device, checkpoint=checkpoint, training=False)
        model.eval()
        tokenizer = AutoTokenizer.from_pretrained(str(checkpoint / "tokenizer"), local_files_only=True)
        native_context = model.config.max_position_embeddings
        tokenizer.model_max_length = native_context
        with torch.inference_mode(), score_path.open("a", encoding="utf-8") as stream:
            for start in range(0, len(missing), args.batch_size):
                batch_rows = missing[start:start + args.batch_size]
                batch = collate_candidates(tokenizer, batch_rows, native_context, device)
                autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
                with autocast:
                    logits = forward_logits(model, batch).float().cpu().tolist()
                for row, logit in zip(batch_rows, logits):
                    record = score_record(row, logit)
                    stream.write(json.dumps(record, allow_nan=False) + "\n")
                    saved[row_key(row)] = record
                stream.flush()
                if start % 100 == 0:
                    print(json.dumps(dict(event="scoring_progress", completed=len(saved), total=len(rows))), flush=True)
    predictions = select_predictions(rows, saved)
    with (output / "predictions.jsonl").open("w", encoding="utf-8") as stream:
        for record in predictions:
            stream.write(json.dumps(record, allow_nan=False) + "\n")
    correct = sum(record["is_correct"] for record in predictions)
    result = dict(status="complete", config=config, correct=correct, questions=len(predictions),
                  accuracy=correct / len(predictions), accuracy_percent=100 * correct / len(predictions),
                  candidate_count=len(saved), elapsed_seconds=time.monotonic() - started)
    write_json(output / "results.json", result)
    print(json.dumps(result), flush=True)
    return result


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center", required=True)
    parser.add_argument("--generator", required=True, help="Target generator ID")
    parser.add_argument("--candidates", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--question-start", type=int, default=0)
    parser.add_argument("--question-end", type=int)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--device", default="cuda")
    return parser


if __name__ == "__main__":
    run_scoring(build_parser().parse_args())
