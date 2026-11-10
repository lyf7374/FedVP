"""Train one local source's eight candidates using the verifier's original head."""
from contextlib import nullcontext
import argparse
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import random
import time

from train_skywork_verifier import (training_defaults as skywork_defaults, configure_environment,
                                    write_json, load_candidate_sources, collate_candidates, epoch_batches)
from heterogeneous_verifier import load_verifier, forward_logits, save_native_head


def training_defaults():
    config = skywork_defaults()
    config.update(model_id=None, verifier_kind="standard", source_count=1, candidates_per_question=8,
                  checkpoint_format="backbone_peft_adapter_plus_native_heads")
    return config


def train_effective_batch(model, tokenizer, rows, optimizer, device, microbatch_size, native_context):
    import torch
    from torch.nn import functional as F

    model.train()
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    for start in range(0, len(rows), microbatch_size):
        batch = collate_candidates(tokenizer, rows[start:start + microbatch_size], native_context, device)
        autocast = torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()
        with autocast:
            # Explicit BCE: HF num_labels=1 otherwise defaults to a regression loss.
            loss = F.binary_cross_entropy_with_logits(forward_logits(model, batch), batch["targets"],
                                                       reduction="sum")
        (loss / len(rows)).backward()
        loss_sum += loss.detach().float().item()
    optimizer.step()
    return loss_sum / len(rows)


def save_checkpoint(model, tokenizer, optimizer, path, config, progress, device):
    import torch

    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    model.model.save_pretrained(path / "adapter", safe_serialization=True, save_embedding_layers=False)
    save_native_head(model, path)
    if tokenizer is not None:
        tokenizer.save_pretrained(path / "tokenizer")
    state = dict(optimizer=optimizer.state_dict(), progress=dict(progress),
                 python_rng=random.getstate(), torch_rng=torch.get_rng_state())
    if device.type == "cuda":
        state["cuda_rng"] = torch.cuda.get_rng_state(device)
    torch.save(state, path / "training_state.pt")
    write_json(path / "manifest.json", dict(config=config, progress=progress,
               saved_at=datetime.now(timezone.utc).isoformat(),
               checkpoint_format="backbone_peft_adapter_plus_native_heads"))


def restore_training_state(path, optimizer, device):
    import torch

    # This is the script's own resumable optimizer/RNG checkpoint, not model weights.
    state = torch.load(Path(path) / "training_state.pt", map_location="cpu", weights_only=False)
    optimizer.load_state_dict(state["optimizer"])
    random.setstate(state["python_rng"])
    torch.set_rng_state(state["torch_rng"])
    if device.type == "cuda" and "cuda_rng" in state:
        torch.cuda.set_rng_state(state["cuda_rng"], device)
    return state["progress"]


def run_training(args):
    configure_environment()
    import torch
    import transformers
    import peft
    from transformers import AutoTokenizer

    config = training_defaults()
    config.update(model_id=args.model_id, verifier_kind=args.verifier_kind)
    for key in ("epochs", "learning_rate", "effective_batch_size", "microbatch_size", "rank",
                "alpha", "dropout", "seed", "checkpoint_steps"):
        config[key] = getattr(args, key)
    for key in ("epochs", "effective_batch_size", "microbatch_size", "rank", "checkpoint_steps"):
        if config[key] < 1:
            raise ValueError(f"{key} must be positive")
    if config["learning_rate"] <= 0 or config["alpha"] <= 0 or not 0 <= config["dropout"] < 1:
        raise ValueError("learning_rate/alpha must be positive and dropout must be in [0,1)")
    if config["microbatch_size"] > config["effective_batch_size"]:
        raise ValueError("microbatch_size cannot exceed effective_batch_size")
    if len(args.candidates) != 1:
        raise ValueError("This experiment uses exactly one assigned local generator source")
    rows, sources = load_candidate_sources(args.candidates, args.center, "train")
    from collections import Counter
    if set(Counter(row["sample_idx"] for row in rows).values()) != {8}:
        raise ValueError("Every local training question must contain all eight candidates")
    output = Path(args.output_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    config.update(center=args.center, sources=sources, model_path=str(Path(args.model_path).resolve()),
                  output_dir=str(output), candidate_count=len(rows),
                  source_question_count=sum(source["question_count"] for source in sources),
                  device=str(device), package_versions=dict(torch=torch.__version__,
                  transformers=transformers.__version__, peft=peft.__version__))
    if device.type == "cpu":
        config.update(backbone_dtype="float32", compute_dtype="float32")
    resume = Path(args.resume).resolve() if args.resume else None
    if resume:
        saved = json.loads((resume / "manifest.json").read_text(encoding="utf-8"))["config"]
        for key in training_defaults().keys() | {"center", "sources", "model_path"}:
            if saved[key] != config[key]:
                raise ValueError(f"Resume setting differs: {key}")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type == "cuda":
        torch.cuda.manual_seed(config["seed"])
    model = load_verifier(config["model_path"], config, device, checkpoint=resume)
    tokenizer_path = resume / "tokenizer" if resume else config["model_path"]
    tokenizer = AutoTokenizer.from_pretrained(str(tokenizer_path), local_files_only=True, use_fast=True)
    native_context = model.config.max_position_embeddings
    tokenizer.model_max_length = native_context
    config["native_context"] = native_context
    config["trainable_parameters"] = sum(parameter.numel() for parameter in model.parameters()
                                         if parameter.requires_grad)
    optimizer = torch.optim.AdamW((parameter for parameter in model.parameters() if parameter.requires_grad),
                                  lr=config["learning_rate"], betas=tuple(config["optimizer_betas"]),
                                  eps=config["optimizer_eps"], weight_decay=config["weight_decay"])
    progress = dict(epoch=0, next_batch=0, optimizer_steps=0)
    if resume:
        progress = restore_training_state(resume, optimizer, device)
    write_json(output / "training_config.json", config)
    log_path = output / "training_log.jsonl"
    if log_path.exists():
        if not resume:
            raise ValueError("Output already contains a training log; use --resume for this run")
        retained = [line for line in log_path.read_text(encoding="utf-8").splitlines()
                    if json.loads(line)["optimizer_step"] <= progress["optimizer_steps"]]
        log_path.write_text("".join(line + "\n" for line in retained), encoding="utf-8")
    checkpoint = output / "checkpoint"
    started = time.monotonic()
    updates_per_epoch = math.ceil(len(rows) / config["effective_batch_size"])
    print(json.dumps(dict(event="training_start", config=config, progress=progress,
                          updates_per_epoch=updates_per_epoch)), flush=True)
    start_epoch = progress["epoch"]
    for epoch in range(start_epoch, config["epochs"]):
        start_batch = progress["next_batch"] if epoch == start_epoch else 0
        for batch_index, indices in enumerate(epoch_batches(len(rows), config["effective_batch_size"],
                                                            config["seed"], epoch)):
            if batch_index < start_batch:
                continue
            batch_rows = [rows[index] for index in indices]
            loss = train_effective_batch(model, tokenizer, batch_rows, optimizer, device,
                                         config["microbatch_size"], native_context)
            progress = dict(epoch=epoch, next_batch=batch_index + 1,
                            optimizer_steps=progress["optimizer_steps"] + 1)
            entry = dict(optimizer_step=progress["optimizer_steps"], epoch=epoch,
                         batch_index=batch_index, candidates=len(batch_rows), bce_loss=loss)
            with log_path.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(entry, allow_nan=False) + "\n")
            print(json.dumps(entry, allow_nan=False), flush=True)
            if progress["optimizer_steps"] % config["checkpoint_steps"] == 0:
                save_checkpoint(model, tokenizer, optimizer, checkpoint, config, progress, device)
        progress.update(epoch=epoch + 1, next_batch=0)
        save_checkpoint(model, tokenizer, optimizer, checkpoint, config, progress, device)
    result = dict(status="complete", checkpoint=str(checkpoint), config=config, progress=progress,
                  elapsed_seconds=time.monotonic() - started, resumed_from=str(resume) if resume else None)
    write_json(output / "training_result.json", result)
    print(json.dumps(result), flush=True)
    return result


def build_parser():
    defaults = training_defaults()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center", required=True)
    parser.add_argument("--candidates", nargs="+", required=True, type=Path)
    parser.add_argument("--model-path", required=True, type=Path)
    parser.add_argument("--model-id", required=True, help="Published verifier repository ID")
    parser.add_argument("--verifier-kind", choices=("standard", "armorm"), default="standard")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--resume", type=Path, help="An existing checkpoint directory")
    parser.add_argument("--device", default="cuda", help="Single worker device; CPU is available for mechanical checks")
    for key in ("epochs", "effective_batch_size", "microbatch_size", "rank", "alpha", "seed", "checkpoint_steps"):
        parser.add_argument("--" + key.replace("_", "-"), type=int, default=defaults[key])
    for key in ("learning_rate", "dropout"):
        parser.add_argument("--" + key.replace("_", "-"), type=float, default=defaults[key])
    return parser


if __name__ == "__main__":
    run_training(build_parser().parse_args())

