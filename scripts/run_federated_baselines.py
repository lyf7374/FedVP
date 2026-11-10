"""Medical-QA FedIT, FedFFA-LoRA, FLoRA and Zhao FedPrompt integration.

Train serial clients on one visible GPU; predict one complete answer per question.
Method equations and parameter provenance are in configs/baseline_training.json.
This entry point never downloads models and does not modify upstream sources.
"""
from __future__ import annotations

import argparse
import gc
import importlib.util
import json
import math
import os
from functools import lru_cache
from itertools import islice
from pathlib import Path
from datetime import datetime, timezone

ROOT = Path(__file__).resolve().parents[1]
METHODS = ("fedit", "fedffa", "flora", "fedprompt")


@lru_cache(maxsize=None)
def import_author(name, root=ROOT):
    path = ((Path(__file__).parent / "answer_extractor.py") if name == "answer_extractor"
            else Path(root) / "upstream" / "FedCoT" / "src" / f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"baseline_fedcot_{name}", path)
    result = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(result)
    return result


def qa_prompt(row, dataset, root=ROOT):
    return import_author("prompt_loader", root).load_prompts(dataset) + "\n\nQ: " + row["question"]


def supervision_target(row):
    """Retain source explanations; make the final line parseable by FedCoT."""
    letter = row["answer_letter"]
    final = f"#### {letter}. {row['options'][letter]}"
    source = str(row.get("answer") or "").strip()
    if "####" in source:
        rationale = source.rsplit("####", 1)[0].strip()
    else:
        bare = {letter.casefold(), str(row.get("answer_string", "")).strip().casefold(),
                str(row["options"][letter]).strip().casefold(),
                f"{letter}. {row['options'][letter]}".casefold()}
        rationale = "" if source.casefold() in bare else source
    return f"{rationale}\n{final}" if rationale else final


def encode_example(tokenizer, row, dataset, train_on_inputs, context_limit,
                   virtual_tokens=0, root=ROOT):
    user = {"role": "user", "content": qa_prompt(row, dataset, root)}
    prefix = list(tokenizer.apply_chat_template([user], tokenize=True, add_generation_prompt=True))
    ids = list(tokenizer.apply_chat_template(
        [user, {"role": "assistant", "content": supervision_target(row)}],
        tokenize=True, add_generation_prompt=False))
    if len(ids) + virtual_tokens > context_limit:
        raise ValueError(f"Training text needs {len(ids) + virtual_tokens} tokens but native context is {context_limit}")
    if ids[:len(prefix)] != prefix:
        raise ValueError("Chat template assistant prefix does not match the training template")
    labels = ids.copy()
    if not train_on_inputs:
        labels[:len(prefix)] = [-100] * len(prefix)
    return {"input_ids": ids, "attention_mask": [1] * len(ids), "labels": labels}


def aggregate_adapters(method, states, counts):
    """CPU FP32 aggregation; inputs remain untouched.

    FLoRA: A = cat(p_k A_k, dim=0); B = cat(B_k, dim=1).
    FedIT: average A and B independently. FFA: retain common A, average B.
    FedPrompt: average prompt_embeddings. All weights are n_k / sum(n_k).
    """
    import torch
    if method not in METHODS:
        raise ValueError(f"Unknown method {method}")
    if not states or len(states) != len(counts) or any(n <= 0 for n in counts):
        raise ValueError("Aggregation needs one positive sample count per client")
    keys = set(states[0])
    if any(set(state) != keys for state in states):
        raise ValueError("Client adapter parameter names differ")
    weights = [n / sum(counts) for n in counts]
    result = {}
    for key in states[0]:
        tensors = [state[key].detach().cpu().float() for state in states]
        if method == "flora":
            if "lora_A" in key:
                result[key] = torch.cat([weight * tensor for weight, tensor in zip(weights, tensors)], dim=0)
            elif "lora_B" in key:
                result[key] = torch.cat(tensors, dim=1)
            else:
                raise ValueError(f"FLoRA stacking received a non-LoRA tensor: {key}")
        elif method == "fedffa" and "lora_A" in key:
            if any(not torch.equal(tensors[0], tensor) for tensor in tensors[1:]):
                raise ValueError(f"FedFFA requires a fixed common A: {key}")
            result[key] = tensors[0].clone()
        else:
            if method == "fedffa" and "lora_B" not in key:
                raise ValueError(f"FedFFA received a non-LoRA tensor: {key}")
            result[key] = sum(weight * tensor for weight, tensor in zip(weights, tensors))
    return result


def stacked_lora_dimensions(rank, alpha, clients):
    # Scaling remains alpha/rank after concatenating the client rank dimensions.
    return rank * clients, alpha * clients


def set_trainable_parameters(model, method):
    for name, parameter in model.named_parameters():
        if method == "fedprompt":
            trainable = "prompt_encoder" in name
        elif method == "fedffa":
            trainable = "lora_B" in name
        else:
            trainable = "lora_A" in name or "lora_B" in name
        parameter.requires_grad_(trainable)


def accumulation_loss_weights(label_batches):
    """Reproduce one full batch's token-mean CE, including its short tail.

    The causal shift discards the first real label. FedPrompt's virtual labels
    and user prefix are -100, so they contribute no supervised target tokens.
    """
    counts = [int((labels[:, 1:] != -100).sum().item()) for labels in label_batches]
    if not counts or any(count == 0 for count in counts):
        raise ValueError("Every training microbatch must contain a supervised target")
    return [count / sum(counts) for count in counts]


def termination_ids(tokenizer, model_eos=None):
    values = list(model_eos) if isinstance(model_eos, (list, tuple)) else [model_eos]
    values.append(tokenizer.eos_token_id)
    vocab = tokenizer.get_vocab()
    values.extend(vocab.get(token) for token in ("<|eot_id|>", "<|im_end|>"))
    return list(dict.fromkeys(int(token) for token in values if token is not None and token >= 0))


def sampling_plan(dataset_counts, seed=42):
    """Author RandomState/ranges/1024-candidate rule; this baseline has n=1."""
    import numpy as np
    rng = np.random.RandomState(seed)
    return {name: [(float(rng.uniform(.3, .9)), float(rng.uniform(.7, .95)))
                   for _ in range(0, count, 1024)]
            for name, count in dataset_counts.items()}


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    temporary.replace(path)


def configure_environment(root):
    paths = {"HF_HOME": "cache/huggingface", "HF_DATASETS_CACHE": "cache/huggingface/datasets",
             "HUGGINGFACE_HUB_CACHE": "cache/huggingface/hub", "XDG_CACHE_HOME": "cache",
             "TORCH_HOME": "cache/torch", "TORCH_EXTENSIONS_DIR": "cache/torch_extensions",
             "TRITON_CACHE_DIR": "cache/triton", "CUDA_CACHE_PATH": "cache/cuda",
             "TMPDIR": "tmp", "TMP": "tmp", "TEMP": "tmp"}
    for name, relative in paths.items():
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[name] = str(path)
    os.environ.update(HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1", TOKENIZERS_PARALLELISM="false")


def resolved(root, path):
    path = Path(path)
    return path if path.is_absolute() else root / path


def adapter_state(model):
    from peft import get_peft_model_state_dict
    return {name: value.detach().cpu().float().clone()
            for name, value in get_peft_model_state_dict(model, save_embedding_layers=False).items()}


def restore_adapter(model, state):
    from peft import set_peft_model_state_dict
    set_peft_model_state_dict(model, state, adapter_name="default")


def add_adapter(base, method, settings, *, stacked_clients=1):
    import torch
    from peft import LoraConfig, PromptTuningConfig, PromptTuningInit, TaskType, get_peft_model
    if method == "fedprompt":
        adapter_config = PromptTuningConfig(task_type=TaskType.CAUSAL_LM,
                                           num_virtual_tokens=settings["num_virtual_tokens"],
                                           prompt_tuning_init=PromptTuningInit.RANDOM)
    else:
        rank, alpha = stacked_lora_dimensions(settings["lora_rank"], settings["lora_alpha"], stacked_clients)
        adapter_config = LoraConfig(task_type=TaskType.CAUSAL_LM, r=rank, lora_alpha=alpha,
                                    lora_dropout=settings["lora_dropout"],
                                    target_modules=settings["target_modules"], bias="none")
    model = get_peft_model(base, adapter_config)
    # Base stays BF16; trainable adapter tensors and optimizer moments stay FP32.
    for name, parameter in model.named_parameters():
        if "lora_A" in name or "lora_B" in name or "prompt_encoder" in name:
            parameter.data = parameter.data.float()
    if method == "fedffa":
        with torch.no_grad():
            for name, parameter in model.named_parameters():
                if "lora_A" in name:
                    torch.nn.init.normal_(parameter, mean=0., std=1. / settings["lora_rank"])
                elif "lora_B" in name:
                    parameter.zero_()
    set_trainable_parameters(model, method)
    return model


def remove_lora_without_merge(model):
    """Discard a client's adapter without modifying the round's shared base."""
    return model.unload()


def collate_examples(examples, pad_id):
    import torch
    length = max(len(item["input_ids"]) for item in examples)
    return {key: torch.tensor([item[key] + [pad_value] * (length - len(item[key]))
                               for item in examples], dtype=torch.long)
            for key, pad_value in (("input_ids", pad_id), ("attention_mask", 0), ("labels", -100))}


def make_optimizer(parameters, settings):
    import torch
    options = {"lr": settings["learning_rate"], "weight_decay": settings["weight_decay"]}
    if settings["optimizer"] == "sgd":
        return torch.optim.SGD(parameters, momentum=settings.get("momentum", 0.), **options)
    options.update(betas=tuple(settings["adam_betas"]), eps=settings["adam_epsilon"])
    return (torch.optim.AdamW if settings["optimizer"] == "adamw" else torch.optim.Adam)(parameters, **options)


def train_client(model, examples, tokenizer, settings, common, device, client_seed, log_path):
    import torch
    from torch.utils.data import DataLoader
    from transformers import get_scheduler, set_seed
    set_seed(client_seed)
    generator = torch.Generator().manual_seed(client_seed)
    loader = DataLoader(examples, batch_size=common["micro_batch_size"], shuffle=True,
                        generator=generator, drop_last=False, num_workers=0,
                        collate_fn=lambda items: collate_examples(items, tokenizer.pad_token_id))
    parameters = [parameter for parameter in model.parameters() if parameter.requires_grad]
    optimizer = make_optimizer(parameters, settings)
    accumulation = common["gradient_accumulation_steps"]
    steps = math.ceil(len(loader) / accumulation) * common["local_epochs"]
    scheduler = get_scheduler(settings["scheduler"], optimizer, num_warmup_steps=settings["warmup_steps"],
                              num_training_steps=steps)
    model.train()
    model.config.use_cache = False
    optimizer.zero_grad(set_to_none=True)
    optimizer_step = 0
    seen = 0
    summed_loss = 0.
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w", encoding="utf-8") as log:
        for epoch in range(common["local_epochs"]):
            iterator = iter(loader)
            for window_start in range(0, len(loader), accumulation):
                window = list(islice(iterator, accumulation))
                weights = accumulation_loss_weights([batch["labels"] for batch in window])
                for index, (batch, weight) in enumerate(zip(window, weights), start=window_start):
                    batch = {name: value.to(device) for name, value in batch.items()}
                    with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                        loss = model(**batch).loss
                    if not bool(torch.isfinite(loss).item()):
                        raise RuntimeError(f"Non-finite training loss at local epoch {epoch}, microbatch {index}")
                    loss_value = float(loss.detach())
                    (loss * weight).backward()
                    batch_count = int(batch["input_ids"].shape[0])
                    seen += batch_count
                    summed_loss += loss_value * batch_count
                    del batch, loss
                if settings["max_grad_norm"] is not None:
                    torch.nn.utils.clip_grad_norm_(parameters, settings["max_grad_norm"])
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad(set_to_none=True)
                optimizer_step += 1
                if optimizer_step % common["logging_steps"] == 0 or optimizer_step == steps:
                    record = {"optimizer_step": optimizer_step, "examples_seen": seen,
                              "mean_example_loss": summed_loss / seen,
                              "learning_rate": scheduler.get_last_lr()[0]}
                    log.write(json.dumps(record) + "\n")
                    log.flush()
                    print(json.dumps(record), flush=True)
                del window
    # Optimizers are recreated for every local client, as in the upstream Trainer.
    result = {"examples_seen": seen, "optimizer_steps": optimizer_step,
              "mean_example_loss": summed_loss / seen, "seed": client_seed}
    del optimizer, scheduler, loader
    return result


def load_base(path, device):
    import torch
    from transformers import AutoModelForCausalLM
    return AutoModelForCausalLM.from_pretrained(
        str(path), local_files_only=True, torch_dtype=torch.bfloat16,
        attn_implementation="sdpa", device_map={"": str(device)}, low_cpu_mem_usage=True)


def train(args):
    import torch
    import transformers
    import peft
    from transformers import AutoConfig, AutoTokenizer, set_seed
    root = args.root.resolve()
    config = json.loads(resolved(root, args.config).read_text("utf-8"))
    common, settings = config["common"], config["methods"][args.method]
    if common["micro_batch_size"] * common["gradient_accumulation_steps"] != common["effective_batch_size"]:
        raise ValueError("Microbatch and accumulation do not implement the configured effective batch")
    model_path = args.model_path
    if model_path is None:
        registry = json.loads((root / "configs/model_paths.json").read_text("utf-8"))
        model_path = Path(registry[args.backbone])
    model_path = resolved(root, model_path)
    output = resolved(root, args.output) if args.output else root / "outputs/baselines" / args.backbone / args.method
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    if device.type != "cuda":
        raise ValueError("This BF16 execution entry point requires an allocated CUDA GPU")
    tokenizer = AutoTokenizer.from_pretrained(str(model_path), local_files_only=True)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model_config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
    context_limit = int(model_config.max_position_embeddings)
    virtual_tokens = settings.get("num_virtual_tokens", 0)
    datasets, length_info = {}, {}
    data_root = resolved(root, args.data_root)
    for name in config["clients"]:
        rows = json.loads((data_root / name / "train.json").read_text("utf-8"))
        if len(rows) != config["clients"][name]:
            raise ValueError(f"{name}: expected existing split of {config['clients'][name]} rows, found {len(rows)}")
        encoded = []
        for index, row in enumerate(rows):
            try:
                encoded.append(encode_example(tokenizer, row, name, settings["train_on_inputs"],
                                               context_limit, virtual_tokens, root))
            except ValueError as exc:
                raise ValueError(f"{name}/train row {index}: {exc}") from exc
        datasets[name] = encoded
        length_info[name] = {"count": len(encoded), "max_tokens": max(len(x["input_ids"]) for x in encoded),
                             "virtual_tokens": virtual_tokens}
    manifest = {"method": args.method, "backbone": args.backbone, "base_model_path": str(model_path),
                "started_utc": datetime.now(timezone.utc).isoformat(), "status": "training",
                "config": config, "context_limit": context_limit, "training_data": length_info,
                "versions": {"torch": torch.__version__, "transformers": transformers.__version__, "peft": peft.__version__},
                "artifact_type": settings["artifact"], "artifact_directory": "final",
                "backbone_dtype": "bfloat16", "adapter_dtype": "float32", "clients": []}
    write_json(output / "run_manifest.json", manifest)
    set_seed(common["seed"])
    base = load_base(model_path, device)
    base.config.use_cache = False
    if common["gradient_checkpointing"]:
        base.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    if args.method != "flora":
        model = add_adapter(base, args.method, settings)
        global_state = adapter_state(model)
    for round_index in range(common["rounds"]):
        local_states, counts = [], []
        for client_index, (name, examples) in enumerate(datasets.items()):
            client_seed = common["seed"] + round_index * len(datasets) + client_index
            set_seed(client_seed)
            if args.method == "flora":
                model = add_adapter(base, args.method, settings)
            else:
                restore_adapter(model, global_state)
                set_trainable_parameters(model, args.method)
            client_dir = output / f"round_{round_index + 1}" / name
            print(f"{args.method} {args.backbone} round {round_index + 1}/{common['rounds']} client {name}", flush=True)
            stats = train_client(model, examples, tokenizer, settings, common, device, client_seed,
                                 client_dir / "training_log.jsonl")
            state = adapter_state(model)
            local_states.append(state)
            counts.append(len(examples))
            model.save_pretrained(client_dir, safe_serialization=True)
            manifest["clients"].append({"round": round_index + 1, "dataset": name, **stats})
            write_json(output / "run_manifest.json", manifest)
            if args.method == "flora":
                base = remove_lora_without_merge(model)
                del model
            gc.collect()
            torch.cuda.empty_cache()
        global_state = aggregate_adapters(args.method, local_states, counts)
        global_dir = output / f"round_{round_index + 1}" / "global"
        if args.method == "flora":
            model = add_adapter(base, args.method, settings, stacked_clients=len(datasets))
            restore_adapter(model, global_state)
            model.save_pretrained(global_dir, safe_serialization=True)
            base = model.merge_and_unload()
            base.to(dtype=torch.bfloat16)
            del model
        else:
            restore_adapter(model, global_state)
            model.save_pretrained(global_dir, safe_serialization=True)
        del local_states
        gc.collect()
        torch.cuda.empty_cache()
    final = output / "final"
    if args.method == "flora":
        base.gradient_checkpointing_disable()
        base.config.use_cache = True
        base.save_pretrained(final, safe_serialization=True, max_shard_size="4GB")
    else:
        model.save_pretrained(final, safe_serialization=True)
    tokenizer.save_pretrained(final)
    manifest.update(status="complete", completed_utc=datetime.now(timezone.utc).isoformat())
    write_json(output / "run_manifest.json", manifest)
    print(f"Saved {settings['artifact']}: {final}", flush=True)


def generate_answer(model, tokenizer, input_ids, device, context_limit, virtual_tokens,
                    max_new_tokens, temperature, top_p, seed):
    """Retry a token-limit hit with the same random seed and a larger budget."""
    import torch
    from transformers import set_seed
    available = context_limit - len(input_ids) - virtual_tokens
    if available <= 0:
        raise RuntimeError("Question leaves no native context for a complete answer")
    budget = min(max_new_tokens, available)
    ends = termination_ids(tokenizer, model.generation_config.eos_token_id)
    inputs = torch.tensor([input_ids], dtype=torch.long, device=device)
    attention_mask = torch.ones_like(inputs)
    attempted_budgets = []
    while True:
        set_seed(seed)
        attempted_budgets.append(budget)
        options = {"max_new_tokens": budget, "do_sample": temperature > 0,
                   "eos_token_id": ends, "pad_token_id": tokenizer.pad_token_id,
                   "use_cache": True, "return_dict_in_generate": True}
        if temperature > 0:
            options.update(temperature=temperature, top_p=top_p)
        with torch.inference_mode():
            output = model.generate(input_ids=inputs, attention_mask=attention_mask, **options)
        sequence = output.sequences[0]
        # PEFT >=0.15 keeps the original input_ids in the generated sequence.
        if sequence.numel() < len(input_ids) or not torch.equal(sequence[:len(input_ids)], inputs[0]):
            raise RuntimeError("Generation backend did not preserve the input prefix; use the deployed PEFT 0.15.2 environment")
        generated = sequence[len(input_ids):]
        finished = generated.numel() > 0 and int(generated[-1]) in ends
        if finished:
            return {"response_text": tokenizer.decode(generated, skip_special_tokens=True),
                    "num_gen_tokens": int(generated.numel()), "finish_reason": "stop",
                    "max_new_tokens_used": budget, "temperature": temperature, "top_p": top_p, "seed": seed}
        if budget >= available:
            failure = RuntimeError("Generation reached the native context limit without a terminator")
            failure.response = {
                "response_text": tokenizer.decode(generated, skip_special_tokens=True),
                "response_text_with_special_tokens": tokenizer.decode(generated, skip_special_tokens=False),
                "generated_token_ids": generated.detach().cpu().tolist(),
                "num_gen_tokens": int(generated.numel()), "prompt_tokens": len(input_ids),
                "virtual_tokens": virtual_tokens, "native_context_limit": context_limit,
                "finish_reason": "native_context_limit", "max_new_tokens_used": budget,
                "initial_max_new_tokens": max_new_tokens, "attempted_budgets": attempted_budgets,
                "temperature": temperature, "top_p": top_p, "seed": seed,
                "termination_ids": ends,
            }
            raise failure
        budget = min(2 * budget, available)
        print(f"Regenerating incomplete answer with max_new_tokens={budget}", flush=True)


def is_authorized_generation_failure(sample, *, root, method, backbone, dataset, sample_idx,
                                     seed, sampling=None):
    """Accept only the exact classic-FL attempt reviewed under the user's policy."""
    approval_relative = "configs/classic_fl_generation_failures_20261026.json"
    reason = "repetitive_generation_no_final_answer"
    if (root is None or method not in ("fedprox", "scaffold", "fedadam", "fedadagrad")
            or backbone not in ("Meta-Llama-3-8B-Instruct", "Qwen2.5-7B-Instruct")):
        return False
    root = Path(root).resolve()
    try:
        candidate, = sample["candidates"]
        if (sample["sample_idx"] != sample_idx or sample["dataset_name"] != dataset
                or sample["num_candidates"] != 1 or sample["num_correct"] != 0 or sample["accuracy"] != 0
                or candidate["candidate_id"] != 0 or candidate["seed"] != seed
                or candidate["finish_reason"] != "generation_failure"
                or candidate["observed_finish_reason"] != "native_context_limit"
                or candidate["failure_reason"] != reason or candidate["resolution"] != approval_relative
                or candidate["extracted_answer"] != "" or candidate["reward"] != 0
                or candidate["is_correct"] is not False):
            return False
        approval = json.loads((root / approval_relative).read_text("utf-8"))
        if (approval["status"] != "user_approved" or approval["sampling_changed"] is not False
                or approval["include_in_accuracy_denominator"] is not True):
            return False
        identity = dict(method=method, backbone=backbone, dataset=dataset, sample_idx=sample_idx)
        entries = [entry for entry in approval["entries"]
                   if all(entry.get(key) == value for key, value in identity.items())]
        if len(entries) != 1:
            return False
        entry = entries[0]
        if (entry["status"] != "resolved" or entry["confirmed_repetition"] is not True
                or entry["parser_answer"] != "" or entry["failure_reason"] != reason
                or candidate["evidence"] != entry["evidence"]
                or any(candidate[key] != entry[key] for key in ("seed", "temperature", "top_p"))):
            return False
        if sampling is not None and tuple(sampling) != (candidate["temperature"], candidate["top_p"]):
            return False
        evidence_path = (root / entry["evidence"]).resolve()
        evidence_path.relative_to(root)
        evidence = json.loads(evidence_path.read_text("utf-8"))
        if any(evidence.get(key) != value for key, value in identity.items()) or evidence["candidate_id"] != 0:
            return False
        response = evidence["response"]
        if (response["finish_reason"] != "native_context_limit"
                or any(candidate.get(key) != value for key, value in response.items() if key != "finish_reason")
                or response["num_gen_tokens"] <= 0
                or response["num_gen_tokens"] != len(response["generated_token_ids"])
                or response["num_gen_tokens"] != response["max_new_tokens_used"]
                or response["num_gen_tokens"] != response["native_context_limit"] - response["prompt_tokens"] - response["virtual_tokens"]
                or any(token in response["termination_ids"] for token in response["generated_token_ids"])):
            return False
        extractor = import_author("answer_extractor", root)
        if extractor.ANSWER_PARSER_VERSION != "final-answer-v3-2026-10-21":
            return False
        return extractor.match_choice_answer(extractor.extract_answer(response["response_text"]), sample["options"]) == ""
    except (OSError, ValueError, TypeError, KeyError, AttributeError):
        return False


def is_authorized_native_limit_score(sample, *, root, method, backbone, dataset, sample_idx,
                                     seed, sampling=None):
    """Accept only the user's exact existing-answer scoring decision for Qwen773."""
    relative = 'configs/classic_fl_native_limit_resolution_20261027.json'
    identity = dict(method=method, backbone=backbone, dataset=dataset, sample_idx=sample_idx)
    if root is None or (method, backbone, dataset, sample_idx) != (
            'fedadam', 'Qwen2.5-7B-Instruct', 'medmcqa', 773):
        return False
    root = Path(root).resolve()
    try:
        policy = json.loads((root / relative).read_text('utf8'))
        entry, = policy['entries']
        candidate, = sample['candidates']
        if (policy['status'] != 'user_approved' or policy['scope'] != 'exact_entry_only'
                or policy['sampling_changed'] is not False or policy['new_generations'] != 0
                or policy['include_in_accuracy_denominator'] is not True
                or policy['parser_version'] != 'final-answer-v3-2026-10-21'
                or any(entry.get(key) != value for key, value in identity.items())
                or entry['status'] != 'approved_for_scoring' or entry['parser_answer'] != 'D'
                or entry['candidate_id'] != 0
                or sample['sample_idx'] != sample_idx or sample['dataset_name'] != dataset
                or sample['num_candidates'] != 1 or candidate['candidate_id'] != 0
                or candidate['finish_reason'] != 'native_context_limit'
                or candidate.get('generation_failure') or candidate.get('failure_reason')
                or candidate['resolution'] != relative or candidate['evidence'] != entry['evidence']
                or candidate['scoring_basis'] != 'user_approved_existing_parsed_answer'
                or candidate['seed'] != seed or seed != 42 + sample_idx
                or any(candidate[key] != entry[key] for key in ('seed', 'temperature', 'top_p'))):
            return False
        if sampling is not None and tuple(sampling) != (candidate['temperature'], candidate['top_p']):
            return False
        evidence_path = (root / entry['evidence']).resolve()
        evidence_path.relative_to(root)
        evidence = json.loads(evidence_path.read_text('utf8'))
        if any(evidence.get(key) != value for key, value in identity.items()) or evidence['candidate_id'] != 0:
            return False
        row = json.loads((root / 'data/processed' / dataset / 'test.json').read_text('utf8'))[sample_idx]
        if (any(sample.get(key) != row[key] for key in
                ('question', 'answer_string', 'answer_letter', 'answer_idx', 'options'))
                or evidence['question'] != row['question'] or evidence['options'] != row['options']):
            return False
        response = evidence['response']
        if (response['finish_reason'] != 'native_context_limit'
                or any(candidate.get(key) != value for key, value in response.items())
                or response['num_gen_tokens'] <= 0
                or response['num_gen_tokens'] != len(response['generated_token_ids'])
                or response['num_gen_tokens'] != response['max_new_tokens_used']
                or response['num_gen_tokens'] != response['native_context_limit'] - response['prompt_tokens'] - response['virtual_tokens']
                or any(token in response['termination_ids'] for token in response['generated_token_ids'])):
            return False
        extractor = import_author('answer_extractor', root)
        answer = extractor.match_choice_answer(extractor.extract_answer(response['response_text']), row['options'])
        reward = extractor.compute_reward(answer, row)
        return (extractor.ANSWER_PARSER_VERSION == policy['parser_version']
                and answer == entry['parser_answer'] == candidate['extracted_answer']
                and candidate['reward'] == reward and candidate['is_correct'] is bool(reward)
                and sample['num_correct'] == int(reward) and sample['accuracy'] == reward)
    except (OSError, ValueError, TypeError, KeyError, AttributeError, IndexError):
        return False


def pending_prediction_ids(directory, rows, seed, start_index, end_index, *, root=None,
                           method=None, backbone=None, dataset=None, sampling=None):
    """Read attempted, unscored native-limit IDs without deciding their correctness."""
    pending = set()
    for path in (Path(directory) / 'pending_reviews').glob('sample_*.json'):
        if path.name.endswith('.resolution.json'):
            continue
        review = json.loads(path.read_text('utf8'))
        index = review['sample_idx']
        response = review['response']
        if (type(index) is not int or not start_index <= index < end_index
                or path.name != f'sample_{index:05d}.json'
                or review.get('status') != 'pending_review'
                or review.get('method') != method or review.get('backbone') != backbone
                or review.get('dataset') != dataset or review.get('candidate_id') != 0
                or review.get('question') != rows[index]['question']
                or review.get('options') != rows[index]['options']
                or response.get('seed') != seed + index
                or response.get('finish_reason') != 'native_context_limit'
                or (sampling is not None and tuple(sampling[index // 1024]) !=
                    (response.get('temperature'), response.get('top_p')))):
            raise ValueError(f'{path}: pending attempt does not match this original question and sampling')
        pending.add(index)
    return pending


def saved_prediction_samples(progress, rows, seed, start_index=0, end_index=None, *,
                             root=None, method=None, backbone=None, dataset=None, sampling=None,
                             allow_gaps=False):
    end_index = len(rows) if end_index is None else end_index
    if allow_gaps and method not in ('fedprox', 'scaffold', 'fedadam', 'fedadagrad'):
        raise ValueError('Sparse prediction recovery is only enabled for the authorized classic methods')
    samples = []
    seen = set()
    if progress.is_file():
        with progress.open(encoding="utf-8") as source:
            for offset, line in enumerate(source):
                sample = json.loads(line)
                index = sample['sample_idx'] if allow_gaps else start_index + offset
                if (type(index) is not int or not start_index <= index < end_index
                        or sample['sample_idx'] != index or index in seen):
                    raise ValueError(f"{progress}: saved sample indices are not a contiguous prediction range")
                candidate = sample["candidates"][0]
                completed = candidate["finish_reason"] == "stop"
                if not completed:
                    completed = is_authorized_generation_failure(
                        sample, root=root, method=method, backbone=backbone, dataset=dataset,
                        sample_idx=index, seed=seed + index,
                        sampling=sampling[index // 1024] if sampling is not None else None)
                if not completed:
                    completed = is_authorized_native_limit_score(
                        sample, root=root, method=method, backbone=backbone, dataset=dataset,
                        sample_idx=index, seed=seed + index,
                        sampling=sampling[index // 1024] if sampling is not None else None)
                if (sample["question"] != rows[index]["question"]
                        or sample["options"] != rows[index]["options"]
                        or candidate["seed"] != seed + index or not completed):
                    raise ValueError(f"{progress}: saved sample {index} does not match this completed prediction")
                samples.append(sample)
                seen.add(index)
    if allow_gaps and seen:
        pending = pending_prediction_ids(progress.parent, rows, seed, start_index, end_index,
            root=root, method=method, backbone=backbone, dataset=dataset, sampling=sampling)
        if set(range(start_index, max(seen) + 1)).difference(seen, pending):
            raise ValueError(f'{progress}: saved gap has no matching pending attempt')
    return samples


def complete_prediction_samples(dataset_dir, rows, seed, *, root=None, method=None,
                                backbone=None, dataset=None, sampling=None, allow_gaps=False):
    """Combine the untouched prefix and independent ranges only when all IDs exist."""
    context = dict(root=root, method=method, backbone=backbone, dataset=dataset, sampling=sampling,
                   allow_gaps=allow_gaps)
    samples = {sample["sample_idx"]: sample for sample in saved_prediction_samples(
        dataset_dir / "test_predictions.jsonl", rows, seed, **context)}
    for progress in sorted((dataset_dir / "ranges").glob("indices_*_*/test_predictions.jsonl")):
        _, start, end = progress.parent.name.split("_")
        for sample in saved_prediction_samples(progress, rows, seed, int(start), int(end), **context):
            index = sample["sample_idx"]
            if index in samples and samples[index] != sample:
                raise ValueError(f"{progress}: conflicting completed prediction for sample {index}")
            samples[index] = sample
    if set(samples) != set(range(len(rows))):
        return None
    return [samples[index] for index in range(len(rows))]


def finalize_prediction_range(directory, rows, seed, start_index, end_index, *, root=None,
                              method=None, backbone=None, dataset=None, sampling=None,
                              allow_gaps=False):
    """Publish only actual range coverage; can finish a reviewed range entirely on CPU."""
    directory = Path(directory)
    context = dict(root=root, method=method, backbone=backbone, dataset=dataset, sampling=sampling)
    records = saved_prediction_samples(directory / 'test_predictions.jsonl', rows, seed,
        start_index, end_index, allow_gaps=allow_gaps, **context)
    completed = {record['sample_idx'] for record in records}
    expected = set(range(start_index, end_index))
    pending = (pending_prediction_ids(directory, rows, seed, start_index, end_index, **context)
               if allow_gaps else set()).difference(completed)
    unattempted = expected.difference(completed, pending)
    status = 'complete' if completed == expected else ('incomplete' if unattempted else 'pending_review')
    result = dict(status=status, dataset=dataset, start_index=start_index, end_index=end_index,
                  completed_samples=len(completed), pending_indices=sorted(pending),
                  unattempted_indices=sorted(unattempted),
                  dataset_complete=status == 'complete' and start_index == 0 and end_index == len(rows),
                  updated_utc=datetime.now(timezone.utc).isoformat())
    if status == 'complete':
        result['completed_utc'] = result['updated_utc']
    write_json(directory / 'range_status.json', result)
    return result


def predict(args):
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer
    root = args.root.resolve()
    run_dir = resolved(root, args.run_dir)
    manifest = json.loads((run_dir / "run_manifest.json").read_text("utf-8"))
    classic_pending = manifest['method'] in ('fedprox', 'scaffold', 'fedadam', 'fedadagrad')
    if manifest["status"] != "complete":
        raise ValueError("Training artifact is not complete")
    artifact = run_dir / manifest["artifact_directory"]
    settings = manifest["config"]["methods"][manifest["method"]]
    device = torch.device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(str(artifact), local_files_only=True)
    if manifest["artifact_type"] == "merged_hf_model":
        model = load_base(artifact, device)
    else:
        base = load_base(args.model_path or manifest["base_model_path"], device)
        model = PeftModel.from_pretrained(base, str(artifact), is_trainable=False, local_files_only=True)
    model.eval()
    model.config.use_cache = True
    model.gradient_checkpointing_disable()
    extractor = import_author("answer_extractor", root)
    output_dir = resolved(root, args.output) if args.output else run_dir / "predictions"
    data_root = resolved(root, args.data_root)
    summary_path = output_dir / "accuracy.json"
    summaries = json.loads(summary_path.read_text("utf-8")) if summary_path.is_file() else {}
    test_rows = {name: json.loads((data_root / name / "test.json").read_text("utf-8"))
                 for name in manifest["config"]["clients"]}
    sampling = sampling_plan({name: len(rows) for name, rows in test_rows.items()}, args.seed)
    selected = set(args.datasets.split(",")) if args.datasets else set(test_rows)
    if not selected.issubset(test_rows):
        raise ValueError(f"Unknown prediction datasets: {sorted(selected.difference(test_rows))}")
    range_start, range_end = getattr(args, "start_index", None), getattr(args, "end_index", None)
    ranged = range_start is not None or range_end is not None
    if ranged:
        if len(selected) != 1 or range_start is None or range_end is None:
            raise ValueError("Prediction ranges require one dataset and both start_index and end_index")
        selected_name = next(iter(selected))
        if not 0 <= range_start < range_end <= len(test_rows[selected_name]):
            raise ValueError("Prediction range must be within the original test dataset")
        config_dir = (output_dir / selected_name / "ranges"
                      / f"indices_{range_start:05d}_{range_end:05d}")
    else:
        config_dir = output_dir
    prediction_config = {"run_dir": str(run_dir),
               "temperature_range": [.3, .9], "top_p_range": [.7, .95], "candidate_batch_size": 1024,
               "sampling_plan": sampling, "sampling_source": "upstream/FedCoT/src/best_of_n_generator.py; scripts/run_candidates.py",
               "initial_max_new_tokens": args.max_new_tokens,
               "num_candidates": 1, "seed": args.seed, "split": "test", "context_limit": manifest["context_limit"]}
    if ranged:
        prediction_config.update(dataset=selected_name, start_index=range_start, end_index=range_end,
                                 index_scope="original_zero_based_test_indices", end_index_exclusive=True)
    write_json(config_dir / "prediction_config.json", prediction_config)
    for name, rows in test_rows.items():
        if name not in selected:
            continue
        start_index, end_index = (range_start, range_end) if ranged else (0, len(rows))
        dataset_dir = output_dir / name
        progress_dir = (dataset_dir / "ranges" / f"indices_{start_index:05d}_{end_index:05d}"
                        if ranged else dataset_dir)
        progress = progress_dir / "test_predictions.jsonl"
        progress.parent.mkdir(parents=True, exist_ok=True)
        review_context = dict(root=root, method=manifest["method"], backbone=manifest.get("backbone"),
                              dataset=name, sampling=sampling[name])
        samples = saved_prediction_samples(progress, rows, args.seed, start_index, end_index,
                                           allow_gaps=classic_pending, **review_context)
        completed_ids = {sample['sample_idx'] for sample in samples}
        pending_ids = (pending_prediction_ids(progress_dir, rows, args.seed, start_index, end_index,
                                             **review_context) if classic_pending else set())
        resumed_count = len(samples)
        if resumed_count:
            print(f"Resuming {name} [{start_index}:{end_index}]: "
                  f"{resumed_count}/{end_index - start_index} complete answers", flush=True)
        with progress.open("a", encoding="utf-8") as destination:
            for index in range(start_index if classic_pending else start_index + resumed_count, end_index):
                if classic_pending and index in completed_ids | pending_ids:
                    continue
                row = rows[index]
                ids = tokenizer.apply_chat_template([{"role": "user", "content": qa_prompt(row, name, root)}],
                                                    tokenize=True, add_generation_prompt=True)
                try:
                    response = generate_answer(model, tokenizer, ids, device, manifest["context_limit"],
                                               settings.get("num_virtual_tokens", 0), args.max_new_tokens,
                                               *sampling[name][index // 1024], args.seed + index)
                except RuntimeError as error:
                    unresolved = getattr(error, "response", None)
                    if unresolved is not None and unresolved.get("finish_reason") == "native_context_limit":
                        write_json(progress_dir / "pending_reviews" / f"sample_{index:05d}.json", {
                            "status": "pending_review", "reason": "native_context_exhausted_without_terminator",
                            "dataset": name, "sample_idx": index, "candidate_id": 0,
                            "method": manifest["method"], "backbone": manifest.get("backbone"),
                            "run_dir": str(run_dir), "question": row["question"], "options": row["options"],
                            "response": unresolved, "recorded_utc": datetime.now(timezone.utc).isoformat(),
                        })
                        if classic_pending:
                            # Release the completed generation frame and its GPU tensors before continuing.
                            error.__traceback__ = None
                            pending_ids.add(index)
                            continue
                    raise
                answer = extractor.match_choice_answer(extractor.extract_answer(response["response_text"]), row["options"])
                reward = extractor.compute_reward(answer, row)
                candidate = dict(response, candidate_id=0, extracted_answer=answer,
                                 reward=reward, is_correct=bool(reward))
                sample = {key: row[key] for key in ("question", "answer_string", "answer_letter", "answer_idx", "options")}
                sample.update(sample_idx=index, candidates=[candidate], num_candidates=1, dataset_name=name,
                              mode="test", num_correct=int(reward), accuracy=reward)
                samples.append(sample)
                completed_ids.add(index)
                destination.write(json.dumps(sample, ensure_ascii=False) + "\n")
                destination.flush()
                if (index + 1) % 20 == 0 or index + 1 == end_index:
                    print(f"{name}: {index + 1}/{len(rows)} complete answers", flush=True)
        if classic_pending:
            range_status = finalize_prediction_range(progress_dir, rows, args.seed, start_index, end_index,
                                                     allow_gaps=True, **review_context)
            if range_status['status'] != 'complete':
                raise RuntimeError(f"{name}: original range remains pending review for indices {range_status['pending_indices']}")
        samples = complete_prediction_samples(dataset_dir, rows, args.seed, allow_gaps=classic_pending,
                                             **review_context)
        if ranged and not classic_pending:
            write_json(progress_dir / "range_status.json", {
                "status": "complete", "dataset": name, "start_index": start_index,
                "end_index": end_index, "completed_samples": end_index - start_index,
                "dataset_complete": samples is not None})
        if samples is None:
            print(f"{name}: dataset pending; completed range saved to {progress_dir}", flush=True)
            continue
        completed_path = output_dir / name / "test_best_of_n.json"
        if resumed_count < end_index - start_index or not completed_path.is_file():
            write_json(completed_path, samples)
        correct = sum(sample["num_correct"] for sample in samples)
        summaries[name] = {"correct": correct, "total": len(samples), "accuracy": correct / len(samples)}
        write_json(summary_path, summaries)
    print(f"Saved predictions: {output_dir}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    trainer = subcommands.add_parser("train", help="Train all five clients for the configured two rounds")
    predictor = subcommands.add_parser("predict", help="Generate one complete answer per test question")
    for command in (trainer, predictor):
        command.add_argument("--root", type=Path, default=ROOT)
        command.add_argument("--data-root", type=Path, default=Path("data/processed"))
        command.add_argument("--model-path", type=Path)
        command.add_argument("--device", default="cuda:0")
        command.add_argument("--output", type=Path)
    trainer.add_argument("--method", choices=METHODS, required=True)
    trainer.add_argument("--backbone", choices=("Meta-Llama-3-8B-Instruct", "Qwen2.5-7B-Instruct"), required=True)
    trainer.add_argument("--config", type=Path, default=Path("configs/baseline_training.json"))
    predictor.add_argument("--run-dir", type=Path, required=True)
    predictor.add_argument("--max-new-tokens", type=int, default=4096)
    predictor.add_argument("--seed", type=int, default=42)
    predictor.add_argument("--datasets", help="Comma-separated test datasets; sampling uses the original full client order")
    predictor.add_argument("--start-index", type=int, help="Inclusive original test index; requires one dataset")
    predictor.add_argument("--end-index", type=int, help="Exclusive original test index; range output is stored independently")
    args = parser.parse_args()
    configure_environment(args.root.resolve())
    (train if args.command == "train" else predict)(args)


if __name__ == "__main__":
    main()
