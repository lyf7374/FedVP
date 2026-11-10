"""Classical FL baselines on the existing medical QA protocol.

Train one method/backbone on one visible GPU. Completed clients and rounds are
reused on restart; an interrupted client restarts deterministically from that
round's global adapter. Algorithm state is persisted at every round boundary.
"""
from __future__ import annotations

import argparse
import copy
from datetime import datetime, timezone
import gc
from itertools import islice
import json
import math
from pathlib import Path

import run_federated_baselines as base

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("fedprox", "fedadam", "fedadagrad")
PARSER_VERSION = "final-answer-v3-2026-10-21"


def now():
    return datetime.now(timezone.utc).isoformat()


def clone_state(state):
    return {key: tensor.detach().cpu().float().clone() for key, tensor in state.items()}


def weighted_delta(global_state, local_states, counts):
    average = base.aggregate_adapters("fedit", local_states, counts)
    if set(average) != set(global_state):
        raise ValueError("Global and local adapter keys differ")
    return {key: average[key] - global_state[key] for key in global_state}


def initial_algorithm_state(method, global_state, clients, settings):
    import torch
    zeros = lambda: {key: torch.zeros_like(value) for key, value in global_state.items()}
    if method == "scaffold":
        return {"global_control": zeros(), "client_controls": {name: zeros() for name in clients}}
    if method in ("fedadam", "fedadagrad"):
        state = {"v": {key: torch.full_like(value, settings["tau"] ** 2)
                       for key, value in global_state.items()}}
        if method == "fedadam":
            state["m"] = zeros()
        return state
    if method != "fedprox":
        raise ValueError(f"Unknown method {method}")
    return {}


def server_update(method, global_state, local_states, counts, history, settings, new_controls=None):
    """Paper FedOpt equations; sample-weighted client delta and persistent history."""
    delta = weighted_delta(global_state, local_states, counts)
    result = copy.deepcopy(history)
    if method in ("fedprox", "scaffold"):
        updated = base.aggregate_adapters("fedit", local_states, counts)
        if method == "scaffold":
            old_controls = history["client_controls"]
            if set(new_controls or {}) != set(old_controls):
                raise ValueError("SCAFFOLD needs all five new client controls")
            size = len(old_controls)
            result["global_control"] = {
                key: history["global_control"][key] + sum(
                    new_controls[name][key] - old_controls[name][key] for name in old_controls) / size
                for key in global_state}
            result["client_controls"] = copy.deepcopy(new_controls)
        return updated, result
    updated = {}
    for key, change in delta.items():
        if method == "fedadam":
            result["m"][key] = settings["beta1"] * history["m"][key] + (1 - settings["beta1"]) * change
            result["v"][key] = settings["beta2"] * history["v"][key] + (1 - settings["beta2"]) * change.square()
            direction = result["m"][key]
        elif method == "fedadagrad":
            result["v"][key] = history["v"][key] + change.square()
            direction = change
        else:
            raise ValueError(f"Unknown method {method}")
        updated[key] = global_state[key] + settings["server_lr"] * direction / (
            result["v"][key].sqrt() + settings["tau"])
    return updated, result


def proximal_loss(parameters, global_state, mu):
    return sum((parameter - global_state[key]).square().sum() for key, parameter in parameters.items()) * (mu / 2)


def apply_scaffold_correction(parameters, global_control, client_control, learning_rate):
    import torch
    with torch.no_grad():
        for key, parameter in parameters.items():
            parameter.add_(global_control[key] - client_control[key], alpha=-learning_rate)


def scaffold_client_control(global_state, local_state, old_control, global_control, optimizer_steps, learning_rate):
    if optimizer_steps <= 0 or learning_rate <= 0:
        raise ValueError("SCAFFOLD control update requires actual positive steps and learning rate")
    return {key: old_control[key] - global_control[key] +
            (global_state[key] - local_state[key]) / (optimizer_steps * learning_rate)
            for key in global_state}


def save_checkpoint(path, payload):
    import torch
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def load_checkpoint(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def completed_client(client_dir, round_number, dataset):
    marker = client_dir / "complete.json"
    if not marker.is_file():
        return None
    identity = json.loads(marker.read_text("utf-8"))
    if identity.get("round") != round_number or identity.get("dataset") != dataset:
        raise ValueError("Completed client identity does not match this run")
    payload = load_checkpoint(client_dir / "client_state.pt")
    if payload.get("round") != round_number or payload.get("dataset") != dataset:
        raise ValueError("Client checkpoint identity does not match completion marker")
    return payload


def require_current_parser(root):
    if getattr(base.import_author("answer_extractor", root), "ANSWER_PARSER_VERSION", None) != PARSER_VERSION:
        raise ValueError(f"This run requires {PARSER_VERSION}")


def train_client(model, examples, tokenizer, method, settings, common, device, seed, log_path,
                 global_state, global_control=None, client_control=None):
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_scheduler, set_seed
    set_seed(seed)
    loader = DataLoader(examples, batch_size=common["micro_batch_size"], shuffle=True,
                        generator=torch.Generator().manual_seed(seed), drop_last=False, num_workers=0,
                        collate_fn=lambda rows: base.collate_examples(rows, tokenizer.pad_token_id))
    parameters = {name.replace(".default", ""): parameter for name, parameter in model.named_parameters()
                  if parameter.requires_grad}
    if set(parameters) != set(global_state):
        raise ValueError("Trainable LoRA names differ from the global adapter")
    reference = {key: value.to(device) for key, value in global_state.items()} if method == "fedprox" else None
    global_control_device = {key: value.to(device) for key, value in global_control.items()} if method == "scaffold" else None
    client_control_device = {key: value.to(device) for key, value in client_control.items()} if method == "scaffold" else None
    if method == "scaffold" and (settings["scheduler"] != "constant" or settings["warmup_steps"]):
        raise ValueError("This SCAFFOLD adaptation uses constant local LR without warmup")
    optimizer = base.make_optimizer(list(parameters.values()), settings)
    accumulation = common["gradient_accumulation_steps"]
    total_steps = math.ceil(len(loader) / accumulation) * common["local_epochs"]
    scheduler = get_scheduler(settings["scheduler"], optimizer, num_warmup_steps=settings["warmup_steps"],
                              num_training_steps=total_steps)
    model.train()
    model.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)
    seen, optimizer_steps, summed_loss = 0, 0, 0.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(common["local_epochs"]):
            iterator = iter(loader)
            for window_start in range(0, len(loader), accumulation):
                window = list(islice(iterator, accumulation))
                weights = base.accumulation_loss_weights([batch["labels"] for batch in window])
                for index, (batch, weight) in enumerate(zip(window, weights), start=window_start):
                    batch = {key: value.to(device) for key, value in batch.items()}
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
                        loss = model(**batch).loss
                    if not bool(torch.isfinite(loss).item()):
                        raise RuntimeError(f"Non-finite CE at epoch {epoch}, microbatch {index}")
                    loss_value = float(loss.detach())
                    (loss * weight).backward()
                    count = int(batch["input_ids"].shape[0])
                    seen += count
                    summed_loss += loss_value * count
                    del batch, loss
                prox_value = None
                if method == "fedprox":
                    penalty = proximal_loss(parameters, reference, settings["prox_mu"])
                    if not bool(torch.isfinite(penalty).item()):
                        raise RuntimeError("Non-finite proximal penalty")
                    prox_value = float(penalty.detach())
                    penalty.backward()
                    del penalty
                if settings["max_grad_norm"] is not None:
                    torch.nn.utils.clip_grad_norm_(list(parameters.values()), settings["max_grad_norm"])
                actual_lr = optimizer.param_groups[0]["lr"]
                optimizer.step()
                if method == "scaffold":
                    apply_scaffold_correction(parameters, global_control_device, client_control_device, actual_lr)
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_steps += 1
                if optimizer_steps % common["logging_steps"] == 0 or optimizer_steps == total_steps:
                    record = {"optimizer_step": optimizer_steps, "examples_seen": seen,
                              "mean_example_loss": summed_loss / seen, "learning_rate": actual_lr,
                              "timestamp_utc": now()}
                    if prox_value is not None:
                        record["proximal_penalty"] = prox_value
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps(record), flush=True)
                del window
    result = {"examples_seen": seen, "optimizer_steps": optimizer_steps,
              "mean_example_loss": summed_loss / seen, "seed": seed}
    del optimizer, scheduler, loader, reference, global_control_device, client_control_device
    return result


def train(args):
    import torch
    import transformers
    import peft
    from transformers import AutoConfig, AutoTokenizer, set_seed
    root = args.root.resolve()
    config = json.loads(base.resolved(root, args.config).read_text("utf-8"))
    common, settings = config["common"], config["methods"][args.method]
    require_current_parser(root)
    if common["micro_batch_size"] * common["gradient_accumulation_steps"] != common["effective_batch_size"]:
        raise ValueError("Configured microbatch/accumulation does not preserve effective batch")
    model_path = args.model_path
    if model_path is None:
        model_path = Path(json.loads((root / "configs/model_paths.json").read_text("utf-8"))[args.backbone])
    model_path = base.resolved(root, model_path)
    output = base.resolved(root, args.output) if args.output else root / "outputs/classic_fl_20261026" / args.backbone / args.method
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "run_manifest.json"
    manifest = json.loads(manifest_path.read_text("utf-8")) if manifest_path.is_file() else None
    if manifest:
        if manifest["config"] != config or manifest["method"] != args.method or manifest["backbone"] != args.backbone:
            raise ValueError("Existing run manifest configuration differs; refusing to mix runs")
        if manifest["status"] == "complete":
            if not (output / "final/adapter_model.safetensors").is_file():
                raise ValueError("Completed run is missing its final adapter")
            print(f"Already complete: {output}", flush=True)
            return
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("Training requires an allocated CUDA device")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    context_limit = int(AutoConfig.from_pretrained(str(model_path), local_files_only=True).max_position_embeddings)
    datasets, lengths = {}, {}
    for dataset, expected in config["clients"].items():
        rows = json.loads((base.resolved(root, args.data_root) / dataset / "train.json").read_text("utf-8"))
        if len(rows) != expected:
            raise ValueError(f"{dataset}: expected {expected}, found {len(rows)}")
        datasets[dataset] = []
        for index, row in enumerate(rows):
            try:
                datasets[dataset].append(base.encode_example(tokenizer, row, dataset, settings["train_on_inputs"], context_limit, root=root))
            except ValueError as error:
                raise ValueError(f"{dataset}/train index {index}: {error}") from error
        lengths[dataset] = {"count": expected, "max_tokens": max(len(row["input_ids"]) for row in datasets[dataset]), "virtual_tokens": 0}
    if manifest is None:
        manifest = {"method": args.method, "backbone": args.backbone, "base_model_path": str(model_path),
                    "started_utc": now(), "status": "training", "config": config, "context_limit": context_limit,
                    "training_data": lengths, "versions": {"torch": str(torch.__version__),
                    "transformers": transformers.__version__, "peft": peft.__version__},
                    "artifact_type": "peft_adapter", "artifact_directory": "final", "backbone_dtype": "bfloat16",
                    "adapter_dtype": "float32", "answer_parser_version": PARSER_VERSION, "clients": []}
    else:
        manifest.update(status="training", resumed_utc=now(), base_model_path=str(model_path))
    base.write_json(manifest_path, manifest)
    set_seed(common["seed"])
    backbone = base.load_base(model_path, device)
    backbone.config.use_cache = False
    if common["gradient_checkpointing"]:
        backbone.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model = base.add_adapter(backbone, args.method, settings)
    initial_file = output / "initial_state.pt"
    if initial_file.is_file():
        initial = load_checkpoint(initial_file)
    else:
        initial = {"global_state": base.adapter_state(model)}
        initial["algorithm_state"] = initial_algorithm_state(args.method, initial["global_state"], list(datasets), settings)
        save_checkpoint(initial_file, initial)
    global_state, history = initial["global_state"], initial["algorithm_state"]
    completed_stats = []
    for round_index in range(common["rounds"]):
        round_number = round_index + 1
        round_dir = output / f"round_{round_number}"
        checkpoint = round_dir / "round_checkpoint.pt"
        if checkpoint.is_file():
            payload = load_checkpoint(checkpoint)
            if payload["round"] != round_number:
                raise ValueError("Round checkpoint identity mismatch")
            global_state, history = payload["global_state"], payload["algorithm_state"]
            completed_stats = payload["completed_clients"]
            print(f"Reusing complete round {round_number}", flush=True)
            continue
        states, counts, controls = [], [], {}
        for client_index, (dataset, examples) in enumerate(datasets.items()):
            client_dir = round_dir / dataset
            client_seed = common["seed"] + round_index * len(datasets) + client_index
            payload = completed_client(client_dir, round_number, dataset)
            if payload is None:
                base.restore_adapter(model, global_state)
                base.set_trainable_parameters(model, args.method)
                print(f"{args.method} {args.backbone} round {round_number}/{common['rounds']} client {dataset}", flush=True)
                stats = train_client(model, examples, tokenizer, args.method, settings, common, device, client_seed,
                    client_dir / "training_log.jsonl", global_state,
                    history.get("global_control"), history.get("client_controls", {}).get(dataset))
                state = base.adapter_state(model)
                payload = {"round": round_number, "dataset": dataset, "state": state, "stats": stats}
                if args.method == "scaffold":
                    payload["control"] = scaffold_client_control(global_state, state, history["client_controls"][dataset],
                        history["global_control"], stats["optimizer_steps"], settings["learning_rate"])
                model.save_pretrained(client_dir, safe_serialization=True)
                save_checkpoint(client_dir / "client_state.pt", payload)
                base.write_json(client_dir / "complete.json", {"round": round_number, "dataset": dataset, "completed_utc": now(), **stats})
            else:
                print(f"Reusing complete round {round_number} client {dataset}", flush=True)
            states.append(payload["state"])
            counts.append(len(examples))
            if args.method == "scaffold":
                controls[dataset] = payload["control"]
            completed_stats.append({"round": round_number, "dataset": dataset, **payload["stats"]})
            manifest["clients"] = completed_stats
            manifest["current_round"] = round_number
            base.write_json(manifest_path, manifest)
            gc.collect()
            torch.cuda.empty_cache()
        global_state, history = server_update(args.method, global_state, states, counts, history, settings, controls)
        base.restore_adapter(model, global_state)
        model.save_pretrained(round_dir / "global", safe_serialization=True)
        save_checkpoint(checkpoint, {"round": round_number, "global_state": global_state,
                                    "algorithm_state": history, "completed_clients": completed_stats})
        del states
        gc.collect()
        torch.cuda.empty_cache()
    base.restore_adapter(model, global_state)
    model.save_pretrained(output / "final", safe_serialization=True)
    tokenizer.save_pretrained(output / "final")
    manifest.update(status="complete", clients=completed_stats, completed_utc=now())
    base.write_json(manifest_path, manifest)
    print(f"Saved final adapter: {output / 'final'}", flush=True)


def predict(args):
    require_current_parser(args.root.resolve())
    base.predict(args)


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    trainer = subcommands.add_parser("train")
    predictor = subcommands.add_parser("predict")
    for command in (trainer, predictor):
        command.add_argument("--root", type=Path, default=ROOT)
        command.add_argument("--data-root", type=Path, default=Path("data/processed"))
        command.add_argument("--model-path", type=Path)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--output", type=Path)
    trainer.add_argument("--method", choices=METHODS, required=True)
    trainer.add_argument("--backbone", choices=("Meta-Llama-3-8B-Instruct", "Qwen2.5-7B-Instruct"), required=True)
    trainer.add_argument("--config", type=Path, default=Path("configs/classic_fl_baselines.json"))
    predictor.add_argument("--run-dir", type=Path, required=True)
    predictor.add_argument("--max-new-tokens", type=int, default=4096)
    predictor.add_argument("--seed", type=int, default=42)
    predictor.add_argument("--datasets")
    predictor.add_argument("--start-index", type=int)
    predictor.add_argument("--end-index", type=int)
    return parser


def main():
    args = build_parser().parse_args()
    base.configure_environment(args.root.resolve())
    (train if args.command == "train" else predict)(args)


if __name__ == "__main__":
    main()
