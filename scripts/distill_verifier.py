"""Distill the shared pool into one native-head Skywork-Qwen3-8B verifier.

The source manifest is an ordered JSON list of {generator, candidates,
teacher_labels}; relative paths resolve beside that manifest. Each labels JSONL
contains sample_idx, candidate_id and teacher_probability. Public distillation
uses five source entries; a local student uses its center's single source.
"""
import argparse
import json
import math
import random
from itertools import zip_longest
from pathlib import Path

from train_heterogeneous_verifier import (
    load_verifier, restore_training_state, save_checkpoint, train_effective_batch,
    training_defaults,
)
from train_skywork_verifier import configure_environment, epoch_batches, write_json

MODEL_ID="Skywork/Skywork-Reward-V2-Qwen3-8B"

def load_distillation_sources(manifest_path):
    manifest_path=Path(manifest_path).resolve()
    entries=json.loads(manifest_path.read_text(encoding="utf-8"))
    if not isinstance(entries,list) or not entries:
        raise ValueError("Expected an ordered nonempty list of distillation sources")
    names=[entry["generator"] for entry in entries]
    if len(set(names))!=len(names):
        raise ValueError("Each generator must appear once in the source manifest")
    rows,sources=[],[]
    for source_index,entry in enumerate(entries):
        pool=(manifest_path.parent/entry["candidates"]).resolve()
        labels_path=(manifest_path.parent/entry["teacher_labels"]).resolve()
        samples=json.loads(pool.read_text(encoding="utf-8"))
        if not samples or len({s["sample_idx"] for s in samples})!=len(samples):
            raise ValueError("Expected unique, nonempty source question IDs")
        attempts=[]
        for sample in samples:
            if (sample["mode"]!="train" or
                    [c["candidate_id"] for c in sample["candidates"]]!=list(range(8))):
                raise ValueError("Distillation requires the original eight ordered training candidates")
            for candidate in sample["candidates"]:
                if candidate.get("generation_failure") or candidate.get("finish_reason")=="length":
                    raise ValueError("Unfinished training candidates cannot enter distillation")
                attempts.append((sample,candidate))
        with labels_path.open(encoding="utf-8") as stream:
            for attempt,label in zip_longest(attempts,(json.loads(line) for line in stream)):
                if attempt is None or label is None:
                    raise ValueError("Incomplete or extra teacher labels")
                sample,candidate=attempt
                identity=(sample["sample_idx"],candidate["candidate_id"])
                if (label["sample_idx"],label["candidate_id"])!=identity:
                    raise ValueError("Teacher label identity differs from the candidate sequence")
                if "uid" in label and label["uid"]!=sample.get("uid"):
                    raise ValueError("Teacher label UID identity differs")
                if "source_generator" in label and label["source_generator"]!=entry["generator"]:
                    raise ValueError("Teacher generator identity differs")
                value=label["teacher_probability"]
                if type(value) not in (float,int) or not math.isfinite(value) or not 0<=value<=1:
                    raise ValueError("Teacher probability must be finite and in [0,1]")
                if not isinstance(sample["question"],str) or not isinstance(candidate["response_text"],str):
                    raise ValueError("Complete question and response text are required")
                # Gold correctness is intentionally not forwarded to the student.
                rows.append(dict(source_index=source_index,source_generator=entry["generator"],
                    sample_idx=identity[0],candidate_id=identity[1],question=sample["question"],
                    response_text=candidate["response_text"],target=float(value)))
        sources.append(dict(generator=entry["generator"],question_count=len(samples),
            candidates=str(pool),teacher_labels=str(labels_path),
            file_sizes=[pool.stat().st_size,labels_path.stat().st_size]))
    return rows,sources


def run_training(args):
    configure_environment()
    import torch
    from transformers import AutoConfig,AutoTokenizer
    rows,sources=load_distillation_sources(args.sources)
    native=AutoConfig.from_pretrained(args.model_path,local_files_only=True)
    if native.model_type!="qwen3" or native.num_labels!=1:
        raise ValueError("Expected released Qwen3 reward model with its native scalar head")
    config=training_defaults()
    config.update(model_id=MODEL_ID,verifier_kind="standard",center=args.center,
        model_path=str(args.model_path.resolve()),sources=sources,source_count=len(sources),
        native_context=native.max_position_embeddings,teacher_target="teacher_probability",
        loss="binary_cross_entropy_with_logits_on_teacher_probability",
        candidate_count=len(rows),output_dir=str(args.output_dir.resolve()),device=args.device)
    out=args.output_dir.resolve()
    out.mkdir(parents=True,exist_ok=True)
    device=torch.device(args.device)
    if device.type=="cpu":
        config.update(backbone_dtype="float32",compute_dtype="float32")
    resume=args.resume.resolve() if args.resume else None
    if resume:
        old=json.loads((resume/"manifest.json").read_text(encoding="utf-8"))["config"]
        if old!=config:
            raise ValueError("Resume configuration or source identities changed")
    elif list(out.iterdir()):
        raise ValueError("Output already contains files; use --resume with its matching checkpoint")
    random.seed(config["seed"])
    torch.manual_seed(config["seed"])
    if device.type=="cuda":torch.cuda.manual_seed(config["seed"])
    model=load_verifier(config["model_path"],config,device,checkpoint=resume,training=True)
    tokenizer=AutoTokenizer.from_pretrained(resume/"tokenizer" if resume else args.model_path,
                                           local_files_only=True)
    tokenizer.model_max_length=native.max_position_embeddings
    optimizer=torch.optim.AdamW((p for p in model.parameters() if p.requires_grad),
        lr=config["learning_rate"],betas=tuple(config["optimizer_betas"]),
        eps=config["optimizer_eps"],weight_decay=config["weight_decay"])
    progress=dict(epoch=0,next_batch=0,optimizer_steps=0)
    if resume:progress=restore_training_state(resume,optimizer,device)
    write_json(out/"training_config.json",config)
    checkpoint=out/"checkpoint"
    logfile=out/"training_log.jsonl"
    if resume and logfile.exists():
        retained=[line for line in logfile.read_text(encoding="utf-8").splitlines()
                  if json.loads(line)["optimizer_step"]<=progress["optimizer_steps"]]
        logfile.write_text("".join(line+"\n" for line in retained),encoding="utf-8")
    # The initial state is resumable even before the first periodic checkpoint.
    if not resume:save_checkpoint(model,tokenizer,optimizer,checkpoint,config,progress,device)
    first_epoch=progress["epoch"]
    for epoch in range(first_epoch,config["epochs"]):
        skip=progress["next_batch"] if epoch==first_epoch else 0
        for batch_index,ids in enumerate(epoch_batches(len(rows),config["effective_batch_size"],config["seed"],epoch)):
            if batch_index<skip:continue
            loss=train_effective_batch(model,tokenizer,[rows[i] for i in ids],optimizer,device,
                                       config["microbatch_size"],native.max_position_embeddings)
            progress=dict(epoch=epoch,next_batch=batch_index+1,optimizer_steps=progress["optimizer_steps"]+1)
            entry=dict(epoch=epoch,batch_index=batch_index,optimizer_step=progress["optimizer_steps"],
                       candidates=len(ids),soft_bce=loss)
            with logfile.open("a",encoding="utf-8") as stream:stream.write(json.dumps(entry)+"\n")
            print(json.dumps(entry),flush=True)
            if progress["optimizer_steps"]%config["checkpoint_steps"]==0:
                save_checkpoint(model,tokenizer,optimizer,checkpoint,config,progress,device)
        progress.update(epoch=epoch+1,next_batch=0)
        save_checkpoint(model,tokenizer,optimizer,checkpoint,config,progress,device)
    result=dict(status="complete",config=config,checkpoint=str(checkpoint),progress=progress,
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad))
    write_json(out/"training_result.json",result)
    return result


def build_parser():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--center",required=True,help="Student identifier: public or its local center")
    for key in ("sources","model-path","output-dir"):
        parser.add_argument("--"+key,type=Path,required=True)
    parser.add_argument("--resume",type=Path,help="This student's matching optimizer/RNG checkpoint")
    parser.add_argument("--device",default="cuda")
    return parser

if __name__=="__main__":
    run_training(build_parser().parse_args())
