"""Generate the fixed FedCoT candidate pools and evaluate CoT/self-consistency.

Only standard-library modules are imported until main() starts generation.
The caller selects the one GPU used by this model with CUDA_VISIBLE_DEVICES.
"""
import argparse
from collections import Counter
import importlib.util
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
import urllib.error
import urllib.request


DATASETS = ("bioasq", "pubmedqa", "mmlu", "medmcqa", "medqa")
SPLITS = ("train", "test")
NUM_CANDIDATES = 8
CANDIDATE_BATCH_SIZE = 1024
INITIAL_MAX_TOKENS = 4096
SEED = 42


def ordered_responses(choices, expected_count):
    by_index = {choice["index"]: choice for choice in choices}
    if len(choices) != expected_count or set(by_index) != set(range(expected_count)):
        raise RuntimeError("Completion response indices do not match the requested prompts")
    return [{"response_text": by_index[index]["text"],
             "finish_reason": by_index[index]["finish_reason"]}
            for index in range(expected_count)]


def complete_without_truncation(completion, prompts, temperature, top_p,
                                initial_max_tokens, available_context, on_response=None):
    """Retry only length-limited completions at the same sampling settings."""
    if len(prompts) != len(available_context):
        raise ValueError("Each prompt needs its available context budget")
    budgets = [min(initial_max_tokens, int(limit)) for limit in available_context]
    if any(budget <= 0 for budget in budgets):
        raise RuntimeError("Prompt leaves no context for an answer")
    pending = list(range(len(prompts)))
    results = [None] * len(prompts)
    while pending:
        groups = {}
        for index in pending:
            groups.setdefault(budgets[index], []).append(index)
        retry = []
        for budget, indices in groups.items():
            responses = completion([prompts[index] for index in indices],
                                   temperature, top_p, budget)
            if len(responses) != len(indices):
                raise RuntimeError("Completion count does not match prompt count")
            if on_response is not None:
                # Preserve returned text before a native-context exception can
                # discard this batch. This observer never changes generation.
                for index, response in zip(indices, responses):
                    on_response(index, dict(response), budget)
            for index, response in zip(indices, responses):
                if response["finish_reason"] == "length":
                    if budget >= available_context[index]:
                        raise RuntimeError(
                            f"Generation reached the available context limit for prompt {index}"
                        )
                    budgets[index] = min(budget * 2, available_context[index])
                    retry.append(index)
                else:
                    results[index] = dict(response, max_new_tokens_used=budget,
                                          temperature=temperature, top_p=top_p)
        if retry:
            print(f"Retrying {len(retry)} length-limited candidates with larger budgets", flush=True)
        pending = retry
    return results


def majority_answer(answers):
    valid = [answer for answer in answers if answer is not None and str(answer).strip()]
    counts = Counter(valid)
    return max(counts, key=counts.get) if counts else None


def make_sample(row, sample_idx, dataset_name, mode, candidates):
    result = {name: row[name] for name in
              ("question", "answer_string", "answer_letter", "answer_idx", "options")}
    num_correct = sum(bool(candidate["is_correct"]) for candidate in candidates)
    result.update(sample_idx=sample_idx, candidates=candidates,
                  num_candidates=len(candidates), dataset_name=dataset_name, mode=mode,
                  num_correct=num_correct,
                  accuracy=num_correct / len(candidates) if candidates else 0.0)
    return result


def load_progress(path, repair_tail=True):
    """Restore completed batches; discard an interrupted final JSONL write."""
    completed = {}
    if not path.exists():
        return completed
    valid_end = 0
    with path.open("rb") as source:
        for line in source:
            if not line.endswith(b"\n"):
                break
            batch = json.loads(line)
            for record in batch["records"]:
                completed[(record["sample_idx"], record["candidate_id"])] = record
            valid_end = source.tell()
    if repair_tail and path.stat().st_size != valid_end:
        with path.open("r+b") as destination:
            destination.truncate(valid_end)
    return completed


def write_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + f".{os.getpid()}.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def evaluate_baselines(samples):
    selections = []
    for sample in samples:
        candidates = sample["candidates"]
        cot = candidates[0]
        answer = majority_answer([candidate["extracted_answer"] for candidate in candidates])
        sc = next((candidate for candidate in candidates
                   if answer is not None and candidate["extracted_answer"] == answer), None)
        selections.append({
            "sample_idx": sample["sample_idx"], "answer_letter": sample["answer_letter"],
            "cot_candidate_id": cot["candidate_id"], "cot_answer": cot["extracted_answer"],
            "cot_correct": bool(cot["is_correct"]),
            "sc_candidate_id": sc["candidate_id"] if sc else None,
            "sc_answer": answer, "sc_correct": answer == sample["answer_letter"],
        })
    total = len(selections)
    metrics = {method: {"correct": sum(item[f"{method}_correct"] for item in selections),
                        "total": total}
               for method in ("cot", "sc")}
    for values in metrics.values():
        values["accuracy"] = values["correct"] / total if total else 0.0
    return metrics, selections


def import_source(root, name):
    path = ((Path(__file__).parent / "answer_extractor.py") if name == "answer_extractor"
            else root / "upstream" / "FedCoT" / "src" / f"{name}.py")
    spec = importlib.util.spec_from_file_location(f"fedcot_{name}", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def configure_environment(root):
    locations = {
        "HF_HOME": "cache/huggingface", "HF_DATASETS_CACHE": "cache/huggingface/datasets",
        "HUGGINGFACE_HUB_CACHE": "cache/huggingface/hub", "XDG_CACHE_HOME": "cache",
        "PIP_CACHE_DIR": "cache/pip", "TORCH_HOME": "cache/torch",
        "TRITON_CACHE_DIR": "cache/triton", "TORCH_EXTENSIONS_DIR": "cache/torch_extensions",
        "CUDA_CACHE_PATH": "cache/cuda", "MODELSCOPE_CACHE": "cache/modelscope",
        "IPYTHONDIR": "cache/ipython", "NUMBA_CACHE_DIR": "cache/numba",
        "MPLCONFIGDIR": "cache/matplotlib", "TMPDIR": "tmp", "TEMP": "tmp", "TMP": "tmp",
    }
    for key, relative in locations.items():
        path = root / relative
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = str(path)
    os.environ.update(FEDCOT_ROOT=str(root), PYTHONNOUSERSITE="1", PYTHONDONTWRITEBYTECODE="1",
                      TOKENIZERS_PARALLELISM="false", OMP_NUM_THREADS="4",
                      HF_HUB_OFFLINE="1", TRANSFORMERS_OFFLINE="1")
    os.environ["PATH"] = str(Path(sys.executable).parent) + os.pathsep + os.environ.get("PATH", "")


class SGLangServer:
    def __init__(self, root, model_path, model_name, port, context_length):
        self.root = root
        generation_config = json.loads(
            (model_path / "generation_config.json").read_text(encoding="utf-8"))
        eos_ids = generation_config["eos_token_id"]
        self.stop_token_ids = list(dict.fromkeys([eos_ids] if isinstance(eos_ids, int) else eos_ids))
        self.base_url = f"http://127.0.0.1:{port}"
        self.process = None
        self.log = None
        self.log_path = root / "logs" / f"candidates_server_{model_name}_{port}.log"
        self.command = [sys.executable, "-m", "sglang.launch_server",
                        "--model-path", str(model_path), "--host", "127.0.0.1", "--port", str(port),
                        "--tp", "1", "--dp", "1", "--context-length", str(context_length),
                        "--mem-fraction-static", "0.87", "--max-running-requests", "16",
                        "--chunked-prefill-size", "2048", "--random-seed", str(SEED),
                        "--log-level", "warning"]

    def start(self):
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        self.log = self.log_path.open("a", encoding="utf-8")
        print("Starting SGLang: " + " ".join(self.command), flush=True)
        self.process = subprocess.Popen(self.command, cwd=self.root, env=os.environ.copy(),
                                        stdout=self.log, stderr=subprocess.STDOUT, start_new_session=True)
        deadline = time.monotonic() + 1800
        while time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"SGLang exited with {self.process.returncode}; see {self.log_path}")
            try:
                with urllib.request.urlopen(self.base_url + "/health", timeout=5) as response:
                    if response.status == 200:
                        print(f"SGLang ready at {self.base_url}", flush=True)
                        return
            except (urllib.error.URLError, TimeoutError):
                pass
            time.sleep(2)
        raise RuntimeError(f"SGLang did not become ready; see {self.log_path}")

    def complete(self, prompts, temperature, top_p, max_tokens):
        payload = {"model": "default", "prompt": prompts, "temperature": temperature,
                   "top_p": top_p, "max_tokens": max_tokens, "frequency_penalty": 0.0,
                   "stream": False, "stop_token_ids": self.stop_token_ids}
        request = urllib.request.Request(self.base_url + "/v1/completions",
                                         data=json.dumps(payload).encode("utf-8"),
                                         headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=14400) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode('utf-8', errors='replace')
            raise RuntimeError(f'SGLang HTTP {error.code}: {detail}') from error
        return ordered_responses(result["choices"], len(prompts))

    def close(self):
        if self.process is not None:
            try:
                os.killpg(self.process.pid, signal.SIGTERM)
            except ProcessLookupError:
                pass
            try:
                self.process.wait(timeout=30)
            except subprocess.TimeoutExpired:
                os.killpg(self.process.pid, signal.SIGKILL)
                self.process.wait()
        if self.log is not None:
            self.log.close()


def shard_directory(output_dir, batch_start, batch_end):
    return output_dir / "shards" / f"batches_{batch_start:05d}_{batch_end:05d}"


def collect_candidate_progress(output_dir):
    completed = {}
    paths = [output_dir / "candidates.progress.jsonl"]
    paths.extend(sorted((output_dir / "shards").glob("*/candidates.progress.jsonl")))
    for path in paths:
        for identity, record in load_progress(path, repair_tail=False).items():
            if identity in completed and completed[identity] != record:
                derived = {"extracted_answer", "reward", "is_correct"}
                raw_old = {k: v for k, v in completed[identity].items() if k not in derived}
                raw_new = {k: v for k, v in record.items() if k not in derived}
                if raw_old != raw_new:
                    raise RuntimeError(f"Conflicting candidate records for {identity} in {path}")
            completed[identity] = record
    return completed


def restore_captured_responses(progress_dir, progress_path, completed, rows, prompts,
                               extractor, tokenizer, dataset_name, split, batch_settings,
                               context_length, first_batch, end_batch):
    """Commit captured stop responses before retrying an interrupted batch.

    Captured context-exhausted responses stay pending. A larger execution context
    may retry them through the existing budget-expansion path; an unchanged or
    smaller context must not generate the same unresolved attempt again.
    """
    path = progress_dir / "response_attempts.jsonl"
    if not path.exists():
        return 0
    stopped, exhausted = {}, {}
    with path.open("rb") as stream:
        for line in stream:
            if not line.endswith(b"\n"):
                break  # Preserve an interrupted final capture; never rewrite raw text.
            if not line.strip():
                continue
            attempt = json.loads(line)
            sample_idx, candidate_id = attempt["sample_idx"], attempt["candidate_id"]
            identity = sample_idx, candidate_id
            if identity in completed:
                continue
            if not (0 <= sample_idx < len(rows) and 0 <= candidate_id < NUM_CANDIDATES):
                raise ValueError("Captured response has an invalid candidate identity")
            batch_number = (sample_idx * NUM_CANDIDATES + candidate_id) // CANDIDATE_BATCH_SIZE
            if not first_batch <= batch_number < end_batch:
                continue
            temperature, top_p = batch_settings[batch_number]
            if (attempt["dataset_name"] != dataset_name or attempt["split"] != split or
                    attempt["original_batch"] != batch_number or
                    attempt["prompt"] != prompts[sample_idx] or
                    attempt["temperature"] != temperature or attempt["top_p"] != top_p):
                raise ValueError("Captured response does not match the resumed input or sampling settings")
            if attempt["finish_reason"] == "stop":
                stopped.setdefault(identity, attempt)
            elif (attempt["finish_reason"] == "length" and
                  attempt["max_new_tokens"] >= attempt["available_context"]):
                exhausted[identity] = max(exhausted.get(identity, 0), attempt["context_length"])
    recovered = {}
    for identity, attempt in sorted(stopped.items()):
        sample_idx, candidate_id = identity
        answer = extractor.match_choice_answer(extractor.extract_answer(attempt["response_text"]),
                                               rows[sample_idx]["options"])
        reward = extractor.compute_reward(answer, rows[sample_idx])
        record = dict(response_text=attempt["response_text"], finish_reason="stop",
                      max_new_tokens_used=attempt["max_new_tokens"],
                      temperature=attempt["temperature"], top_p=attempt["top_p"],
                      sample_idx=sample_idx, candidate_id=candidate_id,
                      num_gen_tokens=len(tokenizer.encode(attempt["response_text"], add_special_tokens=False)),
                      extracted_answer=answer, reward=reward, is_correct=reward == 1.0)
        recovered.setdefault(attempt["original_batch"], []).append(record)
    if recovered:
        with progress_path.open("a", encoding="utf-8") as destination:
            for batch_number, records in sorted(recovered.items()):
                destination.write(json.dumps({"batch_start": batch_number * CANDIDATE_BATCH_SIZE,
                                              "records": records}, ensure_ascii=False) + "\n")
                destination.flush()
                for record in records:
                    completed[(record["sample_idx"], record["candidate_id"])] = record
    pending = [identity for identity, captured_context in exhausted.items()
               if identity not in completed and captured_context >= context_length]
    if pending:
        raise RuntimeError(
            f"Generation reached the available context limit for {len(pending)} captured candidates; "
            f"raw responses remain pending in {path}. Do not regenerate at this context or "
            "automatically mark them incorrect. At the native context limit, review is required."
        )
    return sum(len(records) for records in recovered.values())


def merge_candidate_shards(output_dir, rows, dataset_name, split):
    completed = collect_candidate_progress(output_dir)
    extractor = import_source(Path(__file__).resolve().parents[1], "answer_extractor")
    expected = {(sample_idx, candidate_id) for sample_idx in range(len(rows))
                for candidate_id in range(NUM_CANDIDATES)}
    missing = expected - completed.keys()
    if missing:
        print(f"Pending {dataset_name}/{split}: {len(missing)} candidates remain", flush=True)
        return False
    samples = []
    for sample_idx, row in enumerate(rows):
        candidates = [{key: value for key, value in completed[(sample_idx, candidate_id)].items()
                       if key != "sample_idx"} for candidate_id in range(NUM_CANDIDATES)]
        for candidate in candidates:
            answer = ("" if candidate.get("finish_reason") == "generation_failure" else
                      extractor.match_choice_answer(extractor.extract_answer(candidate["response_text"]),
                                                    row["options"]))
            reward = extractor.compute_reward(answer, row)
            candidate.update(extracted_answer=answer, reward=reward, is_correct=reward == 1.0)
        samples.append(make_sample(row, sample_idx, dataset_name, split, candidates))
    path = output_dir / f"{split}_best_of_n.json"
    write_json(path, samples)
    if split == "test":
        metrics, selections = evaluate_baselines(samples)
        write_json(output_dir / "baseline_accuracy.json", metrics)
        write_json(output_dir / "baseline_selections.json", selections)
    print(f"Completed {dataset_name}/{split}: {path}", flush=True)
    return True


def run_split(root, output_root, dataset_name, split, rows, tokenizer, context_length,
              server, prompt_module, extractor, batch_settings, batch_start=None, batch_end=None):
    output_dir = output_root / f"{dataset_name}_best_of_n" / split
    output_dir.mkdir(parents=True, exist_ok=True)
    sharded = batch_start is not None or batch_end is not None
    first_batch = 0 if batch_start is None else batch_start
    end_batch = len(batch_settings) if batch_end is None else batch_end
    if sharded and not 0 <= first_batch < end_batch <= len(batch_settings):
        raise ValueError("Batch range must be a nonempty [start,end) within the original split")
    progress_dir = shard_directory(output_dir, first_batch, end_batch) if sharded else output_dir
    progress_dir.mkdir(parents=True, exist_ok=True)
    progress_path = progress_dir / "candidates.progress.jsonl"
    load_progress(progress_path)  # Only this worker may repair its own interrupted write.
    completed = collect_candidate_progress(output_dir)
    prefix = prompt_module.load_prompts(dataset_name)
    prompts = [tokenizer.apply_chat_template(
        [{"role": "user", "content": prefix + "\n\nQ: " + row["question"]}],
        tokenize=False, add_generation_prompt=True) for row in rows]
    restore_captured_responses(progress_dir, progress_path, completed, rows, prompts,
                              extractor, tokenizer, dataset_name, split, batch_settings,
                              context_length, first_batch, end_batch)
    # SGLang 0.4.6 encodes string prompts with special tokens, then reserves
    # one context slot in tp_worker and another in scheduler.
    available = [context_length - len(tokenizer.encode(prompt, add_special_tokens=True)) - 2
                 for prompt in prompts]
    total_candidates = len(rows) * NUM_CANDIDATES
    for batch_number, start in enumerate(range(0, total_candidates, CANDIDATE_BATCH_SIZE)):
        if not first_batch <= batch_number < end_batch:
            continue
        temperature, top_p = batch_settings[batch_number]
        identities = [divmod(index, NUM_CANDIDATES)
                      for index in range(start, min(start + CANDIDATE_BATCH_SIZE, total_candidates))]
        missing = [identity for identity in identities if identity not in completed]
        if not missing:
            continue
        print(f"{dataset_name}/{split} batch {batch_number + 1}/{len(batch_settings)}: "
              f"{len(missing)} candidates, temperature={temperature:.6f}, top_p={top_p:.6f}", flush=True)
        capture_options = {}
        if os.environ.get("FEDCOT_CAPTURE_RESPONSE_ATTEMPTS") == "1":
            def capture_attempt(index, response, budget):
                sample_idx, candidate_id = missing[index]
                attempt = dict(response, sample_idx=sample_idx, candidate_id=candidate_id,
                    dataset_name=dataset_name, split=split, original_batch=batch_number,
                    max_new_tokens=budget, available_context=available[sample_idx],
                    context_length=context_length, temperature=temperature, top_p=top_p,
                    prompt=prompts[sample_idx],
                    captured_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()))
                with (progress_dir / "response_attempts.jsonl").open("a", encoding="utf-8") as stream:
                    stream.write(json.dumps(attempt, ensure_ascii=False) + "\n")
                    stream.flush()
            capture_options["on_response"] = capture_attempt
        responses = complete_without_truncation(
            server.complete, [prompts[index] for index, _ in missing], temperature, top_p,
            INITIAL_MAX_TOKENS, [available[index] for index, _ in missing], **capture_options)
        records = []
        for (sample_idx, candidate_id), response in zip(missing, responses):
            answer = extractor.match_choice_answer(extractor.extract_answer(response["response_text"]),
                                                   rows[sample_idx]["options"])
            reward = extractor.compute_reward(answer, rows[sample_idx])
            record = dict(response, sample_idx=sample_idx, candidate_id=candidate_id,
                          num_gen_tokens=len(tokenizer.encode(response["response_text"], add_special_tokens=False)),
                          extracted_answer=answer, reward=reward, is_correct=reward == 1.0)
            records.append(record)
        with progress_path.open("a", encoding="utf-8") as destination:
            destination.write(json.dumps({"batch_start": start, "records": records}, ensure_ascii=False) + "\n")
            destination.flush()
        for record in records:
            completed[(record["sample_idx"], record["candidate_id"])] = record
        print(f"Saved {dataset_name}/{split}: {len(completed)}/{total_candidates} candidates", flush=True)
    merge_candidate_shards(output_dir, rows, dataset_name, split)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--port", type=int, default=29035)
    parser.add_argument("--datasets", default=",".join(DATASETS))
    parser.add_argument("--splits", default=",".join(SPLITS))
    parser.add_argument("--output-root", type=Path, default=Path("outputs/best_of_n"))
    parser.add_argument("--batch-start", type=int, help="Inclusive original 1024-candidate batch index")
    parser.add_argument("--batch-end", type=int, help="Exclusive original 1024-candidate batch index")
    parser.add_argument("--merge-only", action="store_true",
                        help="Merge completed shard progress into full author JSON without loading a model")
    args = parser.parse_args()
    root = args.root.resolve()
    configure_environment(root)
    requested_datasets = set(args.datasets.split(","))
    requested_splits = set(args.splits.split(","))
    if requested_datasets - set(DATASETS) or requested_splits - set(SPLITS):
        parser.error("Unknown dataset or split")
    sharded = args.batch_start is not None or args.batch_end is not None
    if sharded and (len(requested_datasets) != 1 or len(requested_splits) != 1):
        parser.error("Batch ranges require exactly one dataset and one split")
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    output_root = output_root / args.model_name
    if args.merge_only:
        for dataset in DATASETS:
            for split in SPLITS:
                if dataset in requested_datasets and split in requested_splits:
                    rows = json.loads((root / "data/processed" / dataset / f"{split}.json").read_text("utf-8"))
                    if not merge_candidate_shards(output_root / f"{dataset}_best_of_n" / split,
                                                  rows, dataset, split):
                        raise RuntimeError(f"Cannot export incomplete candidate pool: {dataset}/{split}")
        return
    if args.model_path is None:
        parser.error("--model-path is required for generation")
    import numpy as np
    from transformers import AutoTokenizer
    config = json.loads((args.model_path / "config.json").read_text(encoding="utf-8"))
    context_length = int(config["max_position_embeddings"])
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    prompt_module = import_source(root, "prompt_loader")
    extractor = import_source(root, "answer_extractor")
    rng = np.random.RandomState(SEED)
    jobs = []
    for dataset in DATASETS:
        for split in SPLITS:
            rows = json.loads((root / "data/processed" / dataset / f"{split}.json").read_text("utf-8"))
            settings = [(float(rng.uniform(0.3, 0.9)), float(rng.uniform(0.7, 0.95)))
                        for _ in range(0, len(rows) * NUM_CANDIDATES, CANDIDATE_BATCH_SIZE)]
            if dataset in requested_datasets and split in requested_splits:
                jobs.append((dataset, split, rows, settings))
    server = SGLangServer(root, args.model_path, args.model_name, args.port, context_length)
    run_config_path = output_root / "run_config.json"
    if sharded:
        args.batch_start = 0 if args.batch_start is None else args.batch_start
        args.batch_end = len(jobs[0][3]) if args.batch_end is None else args.batch_end
        run_config_path = shard_directory(
            output_root / f"{jobs[0][0]}_best_of_n" / jobs[0][1],
            args.batch_start, args.batch_end) / "run_config.json"
    metadata = {
        "model_path": str(args.model_path), "model_name": args.model_name, "seed": SEED,
        "num_candidates": NUM_CANDIDATES, "candidate_batch_size": CANDIDATE_BATCH_SIZE,
        "initial_max_new_tokens": INITIAL_MAX_TOKENS, "context_length": context_length,
        "stop_token_ids": server.stop_token_ids,
        "temp_range": [0.3, 0.9], "top_p_range": [0.7, 0.95],
        "datasets": [name for name in DATASETS if name in requested_datasets],
        "splits": [name for name in SPLITS if name in requested_splits],
    }
    if sharded:
        metadata.update(batch_start=args.batch_start, batch_end=args.batch_end,
                        batch_index_scope="original_split_1024_candidate_batches")
    write_json(run_config_path, metadata)
    try:
        server.start()
        for dataset, split, rows, settings in jobs:
            run_split(root, output_root, dataset, split, rows, tokenizer, context_length,
                      server, prompt_module, extractor, settings,
                      batch_start=args.batch_start, batch_end=args.batch_end)
    finally:
        server.close()


if __name__ == "__main__":
    main()
