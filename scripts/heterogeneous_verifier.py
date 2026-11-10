"""Native reward heads + backbone LoRA for the authorized heterogeneous pool.

ArmoRM's scalar is preserved from the public implementation:
https://huggingface.co/RLHFlow/ArmoRM-Llama3-8B-v0.1/blob/main/modeling_custom.py
Its regression layer and gating train; the published reward transform stays frozen.
No Hugging Face automatic regression loss or replacement scalar head is used.
"""
import json
import math
from pathlib import Path

from train_skywork_verifier import collate_candidates

ARMORM_PATTERN = [128009, 128006, 78191, 128007, 271]


def head_names(kind):
    if kind == "standard":
        return ("score",)
    if kind == "armorm":
        return ("regression_layer", "gating")
    raise ValueError(f"Unknown verifier kind: {kind}")


def armorm_model_class():
    """Version-independent loader retaining every official ArmoRM weight name."""
    import torch
    from torch import nn
    from transformers import LlamaModel, LlamaPreTrainedModel

    class GatingNetwork(nn.Module):
        def __init__(self, config):
            super().__init__()
            values = config.to_dict()
            self.temperature = values.get("gating_temperature", 10)
            self.logit_scale = nn.Parameter(torch.ones(1))
            width = config.hidden_size
            layers = []
            for _ in range(values.get("gating_n_hidden", 3)):
                layers.append(nn.Linear(width, values.get("gating_hidden_dim", 1024)))
                width = values.get("gating_hidden_dim", 1024)
            layers.append(nn.Linear(width, config.num_objectives, bias=True))
            self.layers = nn.ModuleList(layers)

        def forward(self, x):
            for index, layer in enumerate(self.layers):
                x = torch.nn.functional.relu(layer(x)) if index < len(self.layers) - 1 else layer(x)
            return torch.softmax(x / self.temperature, dim=1) * self.logit_scale[0]

    class NativeArmoRM(LlamaPreTrainedModel):
        def __init__(self, config):
            super().__init__(config)
            self.model = LlamaModel(config)
            self.num_objectives = config.to_dict().get("num_objectives", 19)
            config.num_objectives = self.num_objectives
            self.regression_layer = nn.Linear(config.hidden_size, self.num_objectives, bias=False)
            self.post_init()
            self.reward_transform_matrix = nn.Parameter(torch.eye(self.num_objectives), requires_grad=False)
            self.gating = GatingNetwork(config)
            self._verifier_kind = "armorm"

    return NativeArmoRM


def save_native_head(model, path):
    import torch
    state = {name: {key: value.detach().cpu() for key, value in getattr(model, name).state_dict().items()}
             for name in head_names(model._verifier_kind)}
    if model._verifier_kind == "armorm":
        state["reward_transform_matrix"] = model.reward_transform_matrix.detach().cpu()
    torch.save(state, Path(path) / "native_heads.pt")


def restore_native_head(model, path):
    import torch
    state = torch.load(Path(path) / "native_heads.pt", map_location="cpu", weights_only=True)
    for name in head_names(model._verifier_kind):
        getattr(model, name).load_state_dict(state[name])
    if model._verifier_kind == "armorm":
        with torch.no_grad():
            model.reward_transform_matrix.copy_(state["reward_transform_matrix"])


def prepare_lora(model, config, device, *, checkpoint=None, training=True):
    import torch
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    model._verifier_kind = config["verifier_kind"]
    model.requires_grad_(False)
    model.model.to(dtype=torch.bfloat16 if device.type == "cuda" else torch.float32)
    if checkpoint is None:
        adapter = LoraConfig(r=config["rank"], lora_alpha=config["alpha"], lora_dropout=config["dropout"],
                             target_modules=config["target_modules"], bias="none",
                             task_type=TaskType.FEATURE_EXTRACTION)
        model.model = get_peft_model(model.model, adapter)
    else:
        model.model = PeftModel.from_pretrained(model.model, str(Path(checkpoint) / "adapter"),
                                               is_trainable=training, autocast_adapter_dtype=True)
        restore_native_head(model, checkpoint)
    for parameter in model.model.parameters():
        if parameter.requires_grad:
            parameter.data = parameter.data.float()
    for name in head_names(model._verifier_kind):
        getattr(model, name).float().requires_grad_(training)
    if model._verifier_kind == "armorm":
        model.reward_transform_matrix.data = model.reward_transform_matrix.data.float()
        model.reward_transform_matrix.requires_grad_(False)
    if not training:
        model.requires_grad_(False)
    model.config.use_cache = False
    if training and config["gradient_checkpointing"]:
        model.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.to(device)
    return model


def load_verifier(model_path, config, device, *, checkpoint=None, training=True):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoConfig
    arguments = dict(torch_dtype=torch.float32, low_cpu_mem_usage=True,
                     attn_implementation=config["attn_implementation"], local_files_only=True,
                     output_loading_info=True)
    if config["verifier_kind"] == "armorm":
        native_config = AutoConfig.from_pretrained(str(model_path), local_files_only=True)
        model, info = armorm_model_class().from_pretrained(str(model_path), config=native_config, **arguments)
    else:
        model, info = AutoModelForSequenceClassification.from_pretrained(str(model_path), **arguments)
        if model.config.num_labels != 1:
            raise ValueError("Expected the published single-output score head")
    if info.get("missing_keys") or info.get("unexpected_keys") or info.get("mismatched_keys"):
        raise ValueError(f"Published weight loading was not exact: {info}")
    return prepare_lora(model, config, device, checkpoint=checkpoint, training=training)


def gating_positions(input_ids):
    positions = []
    for sequence in input_ids.tolist():
        for offset in range(len(sequence) - len(ARMORM_PATTERN), -1, -1):
            if sequence[offset:offset + len(ARMORM_PATTERN)] == ARMORM_PATTERN:
                positions.append(offset)
                break
        else:
            raise ValueError("ArmoRM official assistant-header gating token pattern is missing")
    return positions


def forward_logits(model, batch):
    import torch
    hidden = model.model(input_ids=batch["input_ids"], attention_mask=batch["attention_mask"],
                         use_cache=False, return_dict=True).last_hidden_state
    indices = torch.arange(hidden.shape[0], device=hidden.device)
    positions = torch.arange(hidden.shape[1], device=hidden.device)[None, :]
    last = positions.masked_fill(~batch["attention_mask"].bool(), -1).amax(dim=1)
    with torch.autocast(device_type=hidden.device.type, enabled=False):
        selected = hidden[indices, last].float()
        if model._verifier_kind == "standard":
            return model.score(selected).squeeze(-1)
        rewards = model.regression_layer(selected)
        prompt = hidden[indices, gating_positions(batch["input_ids"])].float()
        weights = model.gating(prompt)
        return ((rewards @ model.reward_transform_matrix) * weights).sum(dim=1)


def question_range(rows, start=0, end=None):
    keys = list(dict.fromkeys((row["source_index"], row["sample_idx"]) for row in rows))
    if end is None:
        end = len(keys)
    if not 0 <= start < end <= len(keys):
        raise ValueError(f"Invalid question index range [{start}, {end}) of {len(keys)}")
    selected = set(keys[start:end])
    return [row for row in rows if (row["source_index"], row["sample_idx"]) in selected]


def row_key(row):
    return row["center"], row["source_index"], row["sample_idx"], row["candidate_id"]


def score_record(row, value):
    result = {key: row[key] for key in ("center", "source_index", "sample_idx", "candidate_id")}
    if row.get("selection_eligible", True) is False:
        if (row.get("generation_failure") is not True or
                row.get("failure_reason") != "repetitive_generation_no_final_answer" or value is not None):
            raise ValueError("Excluded generation failure must have an explicit reason and no verifier logit")
        result.update(logit=None, sigmoid_score=None, is_correct=False,
                      selection_eligible=False, generation_failure=True,
                      failure_reason=row["failure_reason"])
        if "failure_authorization" in row:
            result["failure_authorization"] = row["failure_authorization"]
        return result
    if row.get("generation_failure"):
        raise ValueError("Generation failure cannot be eligible for selection")
    if value is None or not math.isfinite(value):
        raise ValueError("Nonfinite verifier logit")
    score = 1 / (1 + math.exp(-value)) if value >= 0 else math.exp(value) / (1 + math.exp(value))
    return dict(result, logit=value, sigmoid_score=score, is_correct=bool(row["target"]))


def select_predictions(rows, saved):
    """Keep every question while choosing only among its eligible attempts."""
    selected = {}
    questions = set()
    for row in rows:
        key = (row["source_index"], row["sample_idx"])
        questions.add(key)
        if row.get("selection_eligible", True) is False:
            continue
        record = saved[row_key(row)]
        if key not in selected or record["sigmoid_score"] > selected[key]["sigmoid_score"]:
            selected[key] = record
    if questions != set(selected):
        raise ValueError("No eligible candidate for a question; cannot silently change the denominator")
    return list(selected.values())


def read_saved_scores(path, rows):
    path = Path(path)
    if not path.exists():
        return {}
    allowed = {row_key(row): row for row in rows}
    saved = {}
    lines = path.read_text(encoding="utf-8").splitlines(keepends=True)
    complete_lines = []
    for index, line in enumerate(lines):
        try:
            record = json.loads(line)
        except json.JSONDecodeError:
            # A killed writer may leave its final append incomplete.
            if index == len(lines) - 1 and not line.endswith("\n"):
                path.write_text("".join(complete_lines), encoding="utf-8")
                break
            raise
        key = row_key(record)
        if key not in allowed:
            raise ValueError(f"Saved score outside this question range: {key}")
        if key in saved:
            raise ValueError(f"Duplicate saved candidate score: {key}")
        expected = score_record(allowed[key], record["logit"])
        if expected != record:
            raise ValueError(f"Saved score/label differs for candidate: {key}")
        saved[key] = record
        complete_lines.append(line)
        if index == len(lines) - 1 and not line.endswith("\n"):
            with path.open("a", encoding="utf-8") as stream:
                stream.write("\n")
    return saved
