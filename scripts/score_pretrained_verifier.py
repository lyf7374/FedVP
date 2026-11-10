"""Score a released verifier's native head, without local adapters or fine-tuning."""
import argparse
from contextlib import nullcontext
import json
from pathlib import Path
import time
from heterogeneous_verifier import (armorm_model_class, forward_logits, head_names,
    question_range, read_saved_scores, row_key, score_record, select_predictions)
from train_skywork_verifier import (collate_candidates, configure_environment,
    load_candidate_sources, write_json)


def load_pretrained_verifier(model_path, member, device, attn_implementation="sdpa"):
    """Retain the released native head and ArmoRM gating/transform exactly."""
    model_path = Path(model_path)
    if (model_path / "adapter_config.json").exists():
        raise ValueError("A center adapter directory is not a released verifier")
    kind = member["verifier_kind"]
    names = head_names(kind)
    import torch
    from transformers import AutoConfig, AutoModelForSequenceClassification

    arguments = dict(torch_dtype=torch.float32, low_cpu_mem_usage=True,
                     attn_implementation=attn_implementation, local_files_only=True,
                     output_loading_info=True)
    if kind == "armorm":
        native_config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
        model, info = armorm_model_class().from_pretrained(
            str(model_path), config=native_config, **arguments)
    else:
        model, info = AutoModelForSequenceClassification.from_pretrained(str(model_path), **arguments)
        if model.config.num_labels != 1:
            raise ValueError("Expected the released single-output native score head")
    if any(info.get(key) for key in ("missing_keys", "unexpected_keys", "mismatched_keys", "error_msgs")):
        raise ValueError(f"Published weight loading was not exact: {info}")
    if getattr(model, "peft_config", None):
        raise ValueError("Released verifier must not contain a center adapter")
    model._verifier_kind = kind
    model.requires_grad_(False)
    model.model.to(dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    for name in names:
        getattr(model, name).float()
    if kind == "armorm":
        model.reward_transform_matrix.data = model.reward_transform_matrix.data.float()
        model.reward_transform_matrix.requires_grad_(False)
    model.config.use_cache = False
    model.to(device)
    model.eval()
    return model

def prepare_scoring(args):
    if args.batch_size != 1:
        raise ValueError("The paper uses scoring batch-size=1")
    model_path=args.model_path.resolve()
    native=json.loads((model_path/"config.json").read_text(encoding="utf-8"))
    limit=native["max_position_embeddings"]
    all_rows,sources=load_candidate_sources([args.candidates],args.center,args.split,
        allow_generation_failures=True,generator_name=args.generator.rsplit("/",1)[-1])
    rows=question_range(all_rows,args.question_start,args.question_end)
    questions=len({(row["source_index"],row["sample_idx"]) for row in rows})
    return dict(member_center="pretrained",generator=args.generator,center=args.center,
        split=args.split,sources=sources,verifier_model_id=args.model_id,
        verifier_kind=args.verifier_kind,model_path=str(model_path),local_finetuning=False,
        checkpoint=None,score_head="original_published_head",truncation=False,
        question_start=args.question_start,question_end=args.question_start+questions,
        question_count=questions,candidate_count=len(rows),batch_size=1,native_context=limit,
        attn_implementation="sdpa"),rows


def run_scoring(args):
    configure_environment()
    config, rows = prepare_scoring(args)
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    config_path = output / "scoring_config.json"
    score_path = output / "candidate_scores.jsonl"
    if config_path.exists():
        previous = json.loads(config_path.read_text(encoding="utf-8"))
        for key, value in config.items():
            if key not in previous or previous[key] != value:
                raise ValueError(f"Existing score shard belongs to different inputs: {key}")
    elif score_path.exists() and score_path.stat().st_size:
        raise ValueError("Existing score journal has no released-model identity config")
    write_json(config_path, config)
    saved = read_saved_scores(score_path, rows)
    missing = [row for row in rows if row_key(row) not in saved]
    started = time.monotonic()
    excluded = [row for row in missing if row.get("selection_eligible", True) is False]
    if excluded:
        with score_path.open("a", encoding="utf-8") as stream:
            for row in excluded:
                record = score_record(row, None)
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                saved[row_key(row)] = record
            stream.flush()
        missing = [row for row in missing if row.get("selection_eligible", True)]
    if missing:
        import torch
        from transformers import AutoTokenizer
        device = torch.device(args.device)
        model = load_pretrained_verifier(
            config["model_path"], config, device, config["attn_implementation"])
        if model.config.max_position_embeddings != config["native_context"]:
            raise ValueError("Loaded native context differs from the released config")
        tokenizer = AutoTokenizer.from_pretrained(config["model_path"], local_files_only=True)
        tokenizer.model_max_length = config["native_context"]
        with torch.inference_mode(), score_path.open("a", encoding="utf-8") as stream:
            for index, row in enumerate(missing):
                batch = collate_candidates(tokenizer, [row], config["native_context"], device)
                autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
                with autocast:
                    logits = forward_logits(model, batch).float().cpu().tolist()
                if len(logits) != 1:
                    raise ValueError("Native verifier must return one scalar per candidate")
                record = score_record(row, logits[0])
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                saved[row_key(row)] = record
                if index % 100 == 0:
                    print(json.dumps(dict(event="scoring_progress", completed=len(saved), total=len(rows))), flush=True)
    if len(saved) != len(rows):
        raise ValueError("Score shard is incomplete")
    predictions=select_predictions(rows,saved)
    write_json(output/"predictions.json",predictions)
    correct=sum(row["is_correct"] for row in predictions)
    result = dict(status="complete", config=config, correct=correct,
                  accuracy_percent=100*correct/len(predictions), candidate_count=len(saved),
                  question_count=config["question_count"], elapsed_seconds=time.monotonic() - started)
    write_json(output / "results.json", result)
    print(json.dumps(result), flush=True)
    return result

def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    for option in ("center","generator","model-id"):
        parser.add_argument("--"+option,required=True)
    for option in ("candidates","model-path","output-dir"):
        parser.add_argument("--"+option,type=Path,required=True)
    parser.add_argument("--verifier-kind",choices=("standard","armorm"),default="standard")
    parser.add_argument("--split", choices=("train", "test"), default="test")
    parser.add_argument("--question-start",type=int,default=0)
    parser.add_argument("--question-end",type=int)
    parser.add_argument("--batch-size",type=int,default=1)
    parser.add_argument("--device",default="cuda")
    return parser

if __name__=="__main__":
    run_scoring(build_parser().parse_args())
