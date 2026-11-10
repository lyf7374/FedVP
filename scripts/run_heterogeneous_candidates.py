"""Generate the authorized heterogeneous pools using the original candidate engine.

Only the execution adapter changes: model path, tensor parallelism, environment,
and a separate result namespace. Sampling, prompts, eight candidate IDs, complete
response retry and answer extraction come from run_candidates.py.
"""
import argparse
import json
import os
from pathlib import Path
import subprocess
import urllib.error
import urllib.request
import run_candidates as base


class HeterogeneousServer(base.SGLangServer):
    def __init__(self, root, model_path, model_name, port, context_length, tp=1,
                 server_python=None):
        super().__init__(root, model_path, model_name, port, context_length)
        if tp < 1:
            raise ValueError("Tensor-parallel size must be positive")
        self.command[self.command.index("--tp") + 1] = str(tp)
        if server_python is not None:
            self.command[0] = str(server_python)
        self.log_path = root / "logs/generation" / f"server_{model_name}_{port}.log"

    def start(self):
        if "sglang.launch_server" not in self.command:
            return super().start()
        # /health only confirms HTTP startup. Submitting the 1024-prompt batch
        # before SGLang's own warmup completes can starve it until its 600s timeout.
        self.command[self.command.index("--log-level") + 1] = "info"
        offset = self.log_path.stat().st_size if self.log_path.exists() else 0
        super().start()
        deadline = base.time.monotonic() + 1800
        while base.time.monotonic() < deadline:
            if self.process.poll() is not None:
                raise RuntimeError(f"SGLang exited before warmup completed; see {self.log_path}")
            with self.log_path.open("rb") as stream:
                stream.seek(offset)
                if b"The server is fired up and ready to roll!" in stream.read():
                    print("SGLang internal warmup complete", flush=True)
                    return
            base.time.sleep(1)
        raise RuntimeError(f"SGLang internal warmup did not complete; see {self.log_path}")


def vllm_device_indices(visible, listing):
    indices = {uuid.strip(): index.strip() for index, uuid in
               (line.split(",", 1) for line in listing.splitlines() if line.strip())}
    return ",".join(indices[item.strip()] for item in visible.split(","))


class VLLMServer(HeterogeneousServer):
    """Use the existing Blackwell-compatible vLLM environment with identical sampling."""
    def __init__(self, root, model_path, model_name, port, context_length, tp=1,
                 server_python=None):
        super().__init__(root, model_path, model_name, port, context_length, tp, server_python)
        self.command = [self.command[0], "-m", "vllm.entrypoints.openai.api_server",
                        "--model", str(model_path), "--served-model-name", "default",
                        "--host", "127.0.0.1", "--port", str(port),
                        "--tensor-parallel-size", str(tp), "--max-model-len", str(context_length),
                        "--gpu-memory-utilization", "0.87", "--max-num-seqs", "16",
                        "--max-num-batched-tokens", "2048", "--enable-chunked-prefill",
                        "--generation-config", "vllm", "--seed", str(base.SEED)]

    def start(self):
        visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
        old_order = os.environ.get("CUDA_DEVICE_ORDER")
        try:
            # vLLM 0.10.2's NVML adapter parses CUDA_VISIBLE_DEVICES as integers.
            # Convert only the scheduler-assigned UUIDs, preserving their order.
            if visible.startswith("GPU-"):
                listing = subprocess.check_output([
                    "nvidia-smi", "--query-gpu=index,uuid", "--format=csv,noheader"], text=True)
                os.environ["CUDA_VISIBLE_DEVICES"] = vllm_device_indices(visible, listing)
                os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            super().start()
        finally:
            os.environ["CUDA_VISIBLE_DEVICES"] = visible
            if old_order is None:
                os.environ.pop("CUDA_DEVICE_ORDER", None)
            else:
                os.environ["CUDA_DEVICE_ORDER"] = old_order

    def request_payload(self, prompts, temperature, top_p, max_tokens):
        return {"model": "default", "prompt": prompts, "temperature": temperature,
                "top_p": top_p, "max_tokens": max_tokens, "frequency_penalty": 0.0,
                "stream": False, "stop_token_ids": self.stop_token_ids,
                "add_special_tokens": True}

    def complete(self, prompts, temperature, top_p, max_tokens):
        request = urllib.request.Request(
            self.base_url + "/v1/completions",
            data=json.dumps(self.request_payload(prompts, temperature, top_p, max_tokens)).encode("utf-8"),
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=14400) as response:
                result = json.load(response)
        except urllib.error.HTTPError as error:
            detail = error.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"vLLM HTTP {error.code}: {detail}") from error
        return base.ordered_responses(result["choices"], len(prompts))


def merge_candidates_only(output_dir, rows, dataset_name, split, extractor):
    """Reuse author-format candidate assembly without adding baseline experiments."""
    completed = base.collect_candidate_progress(output_dir)
    expected = {(index, candidate) for index in range(len(rows))
                for candidate in range(base.NUM_CANDIDATES)}
    missing = expected - completed.keys()
    if missing:
        print(f"Pending {dataset_name}/{split}: {len(missing)} candidates remain", flush=True)
        return False
    samples = []
    for sample_idx, row in enumerate(rows):
        candidates = [{k: v for k, v in completed[(sample_idx, candidate_id)].items()
                       if k != "sample_idx"} for candidate_id in range(base.NUM_CANDIDATES)]
        for candidate in candidates:
            from generation_failure_policy import candidate_failure_metadata
            failure = candidate_failure_metadata(candidate,
                generator_name=output_dir.parent.parent.name, dataset=dataset_name,
                split=split, sample_idx=sample_idx, allow=True)
            if failure:
                # Retain the attempt and full raw text; selection excludes it.
                candidate.update(failure)
                continue
            if candidate.get("finish_reason") != "stop":
                raise RuntimeError("Incomplete candidate cannot enter a new heterogeneous pool")
            answer = extractor.match_choice_answer(extractor.extract_answer(candidate["response_text"]),
                                                    row["options"])
            reward = extractor.compute_reward(answer, row)
            candidate.update(extracted_answer=answer, reward=reward, is_correct=reward == 1.0)
        samples.append(base.make_sample(row, sample_idx, dataset_name, split, candidates))
    path = output_dir / f"{split}_best_of_n.json"
    base.write_json(path, samples)
    print(f"Completed {dataset_name}/{split}: {path}", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--model-path", type=Path)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--port", type=int, default=29230)
    parser.add_argument("--datasets", "--dataset", default=",".join(base.DATASETS))
    parser.add_argument("--splits", "--split", default="test")
    parser.add_argument("--output-root", type=Path, default=Path("outputs/best_of_n"))
    parser.add_argument("--batch-start", type=int)
    parser.add_argument("--batch-end", type=int)
    parser.add_argument("--merge-only", action="store_true")
    parser.add_argument("--tp", type=int, default=1)
    parser.add_argument("--server-python", type=Path)
    parser.add_argument("--backend", choices=("sglang", "vllm"), default="sglang")
    parser.add_argument("--context-length", type=int,
                        help="Initial execution context; expands to native context on length exhaustion")
    args = parser.parse_args()
    root = args.root.resolve()
    base.configure_environment(root)
    os.environ["FEDCOT_CAPTURE_RESPONSE_ATTEMPTS"] = "1"
    datasets, splits = set(args.datasets.split(",")), set(args.splits.split(","))
    if datasets - set(base.DATASETS) or splits - set(base.SPLITS):
        parser.error("Unknown dataset or split")
    sharded = args.batch_start is not None or args.batch_end is not None
    if sharded and (len(datasets) != 1 or len(splits) != 1):
        parser.error("A shard must name one dataset and one split")
    output_root = args.output_root if args.output_root.is_absolute() else root / args.output_root
    output_root = output_root / args.model_name
    extractor = base.import_source(root, "answer_extractor")
    if extractor.ANSWER_PARSER_VERSION != "final-answer-v3-2026-10-21":
        raise RuntimeError("The authorized candidate parser is final-answer-v3-2026-10-21")
    base.merge_candidate_shards = lambda output, rows, dataset, split: merge_candidates_only(
        output, rows, dataset, split, extractor)
    if args.merge_only:
        for dataset in base.DATASETS:
            for split in base.SPLITS:
                if dataset in datasets and split in splits:
                    rows = json.loads((root / "data/processed" / dataset / f"{split}.json").read_text("utf-8"))
                    if not base.merge_candidate_shards(output_root / f"{dataset}_best_of_n" / split,
                                                       rows, dataset, split):
                        raise RuntimeError(f"Incomplete candidate pool: {dataset}/{split}")
        return
    if args.model_path is None:
        parser.error("--model-path is required for generation")
    import numpy as np
    from transformers import AutoTokenizer
    model_path = args.model_path.resolve()
    config = json.loads((model_path / "config.json").read_text("utf-8"))
    native_context = int(config["max_position_embeddings"])
    context = args.context_length or native_context
    if not 0 < context <= native_context:
        parser.error("Execution context must be positive and no larger than the native context")
    tokenizer = AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    prompts = base.import_source(root, "prompt_loader")
    rng = np.random.RandomState(base.SEED)
    jobs = []
    # Consume the same full, ordered schedule even when running one selected shard.
    for dataset in base.DATASETS:
        for split in base.SPLITS:
            rows = json.loads((root / "data/processed" / dataset / f"{split}.json").read_text("utf-8"))
            settings = [(float(rng.uniform(.3, .9)), float(rng.uniform(.7, .95)))
                        for _ in range(0, len(rows) * base.NUM_CANDIDATES, base.CANDIDATE_BATCH_SIZE)]
            if dataset in datasets and split in splits:
                jobs.append((dataset, split, rows, settings))
    config_path = output_root / "run_config.json"
    if sharded:
        args.batch_start = args.batch_start or 0
        args.batch_end = len(jobs[0][3]) if args.batch_end is None else args.batch_end
        config_path = base.shard_directory(output_root / f"{jobs[0][0]}_best_of_n" / jobs[0][1],
                                           args.batch_start, args.batch_end) / "run_config.json"
    metadata = dict(model_path=str(model_path), model_name=args.model_name, seed=base.SEED,
                    num_candidates=base.NUM_CANDIDATES, candidate_batch_size=base.CANDIDATE_BATCH_SIZE,
                    initial_max_new_tokens=base.INITIAL_MAX_TOKENS, native_context_length=native_context,
                    temp_range=[.3,.9], top_p_range=[.7,.95],
                    datasets=[x for x in base.DATASETS if x in datasets],
                    splits=[x for x in base.SPLITS if x in splits],
                    tp=args.tp, backend=args.backend, parser_version=extractor.ANSWER_PARSER_VERSION,
                    batch_start=args.batch_start, batch_end=args.batch_end,
                    batch_index_scope="original_split_1024_candidate_batches", context_attempts=[])
    while True:
        server_class = HeterogeneousServer if args.backend == "sglang" else VLLMServer
        server = server_class(root, model_path, args.model_name, args.port, context,
                              args.tp, args.server_python)
        metadata.update(context_length=context, stop_token_ids=server.stop_token_ids,
                        server_python=server.command[0])
        metadata["context_attempts"].append(context)
        base.write_json(config_path, metadata)
        try:
            server.start()
            for dataset, split, rows, settings in jobs:
                base.run_split(root, output_root, dataset, split, rows, tokenizer, context,
                               server, prompts, extractor, settings, args.batch_start, args.batch_end)
            break
        except RuntimeError as error:
            exhausted = ("Generation reached the available context limit" in str(error) or
                         "Prompt leaves no context for an answer" in str(error))
            if not exhausted or context >= native_context:
                raise
            context = min(context * 2, native_context)
            print(f"Length exhaustion: restart with context={context}; no truncated answer is scored", flush=True)
        finally:
            server.close()


if __name__ == "__main__":
    main()
