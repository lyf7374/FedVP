"""Generate N=8 only for saved public-reference indices of one dataset.

Retains original source indices, FedCoT prompts, per-original-batch sampling
schedule, context expansion and captured-response journal. No new data split.
"""
import argparse
import json
from pathlib import Path
import sys
import time

import run_candidates as base
from run_heterogeneous_candidates import HeterogeneousServer

RUN = 'public_reference_candidates'


def request_from_ids(spec, dataset):
    indices = [r['original_train_index'] for r in spec['records'] if r['dataset'] == dataset]
    if not indices or indices != sorted(set(indices)) or len(indices) != spec['public_counts'][dataset]:
        raise ValueError('Saved public IDs differ for the selected dataset')
    return {'selected_indices': indices}


def selected_indices(request, start, end, total):
    source = list(range(total)) if request is None else request['selected_indices']
    end = len(source) if end is None else end
    if not 0 <= start < end <= len(source):
        raise ValueError('Question offsets must be a nonempty subset of the fixed input')
    indices = source[start:end]
    if len(set(indices)) != len(indices) or any(not 0 <= i < total for i in indices):
        raise ValueError('Invalid original question indices')
    return indices


def finished(record):
    return record.get('finish_reason') == 'stop'


def assemble_selected(rows, indices, completed, dataset, split, generator_id):
    result = []
    for index in indices:
        candidates = []
        for cid in range(8):
            record = completed.get((index, cid))
            if record is None or not finished(record):
                raise ValueError(f'Incomplete attempt {index}:{cid}')
            candidates.append({k: v for k, v in record.items() if k != 'sample_idx'})
        sample = base.make_sample(rows[index], index, dataset, split, candidates)
        sample.update(generator_id=generator_id, uid=f'{dataset}:{split}:{index}')
        result.append(sample)
    return result


def sampling_settings(root, dataset, split):
    import numpy as np
    rng = np.random.RandomState(42)
    for d in base.DATASETS:
        for s in base.SPLITS:
            n = len(json.loads((root / 'data/processed' / d / (s + '.json')).read_text('utf-8')))
            settings = [(float(rng.uniform(.3, .9)), float(rng.uniform(.7, .95)))
                        for _ in range(0, n * 8, 1024)]
            if (d, s) == (dataset, split):
                return settings
    raise ValueError('Unknown dataset/split')


def restore_public_captures(output, completed, pending, rows, indices, lengths,
                            settings, native, extractor, tokenizer, config):
    """Recover responses captured before an interrupted journal/pending write.

    Complete responses are parsed on CPU. Native-context exhaustion stays pending
    with its original length metadata; it is neither regenerated nor scored.
    """
    path = output / 'response_attempts.jsonl'
    if not path.exists():
        return 0
    expected = {(i, c) for i in indices for c in range(8)}
    stopped, exhausted = {}, {}
    with path.open('rb') as stream:
        for line in stream:
            if not line.endswith(b'\n'):
                break
            if not line.strip():
                continue
            record = json.loads(line)
            key = (record['sample_idx'], record['candidate_id'])
            if key not in expected:
                raise ValueError('Captured public response has unrelated question/candidate identity')
            if key in completed:
                continue
            index, _ = key
            temperature, top_p = settings[index // 128]
            context, budget = record['context_length'], record['max_new_tokens_used']
            available = context - lengths[index] - 2
            if (record['temperature'] != temperature or record['top_p'] != top_p
                    or not 0 < context <= native or not 0 < budget <= available):
                raise ValueError('Captured public response differs from its sampling/context settings')
            if finished(record):
                stopped.setdefault(key, record)
            elif record['finish_reason'] == 'length' and context == native and budget >= available:
                exhausted[key] = record
    recovered = []
    for (index, cid), saved in sorted(stopped.items()):
        if (index, cid) in pending:
            continue
        record = dict(saved)
        answer = extractor.match_choice_answer(extractor.extract_answer(record['response_text']),
                                               rows[index]['options'])
        reward = extractor.compute_reward(answer, rows[index])
        record.update(extracted_answer=answer, reward=reward, is_correct=reward == 1.0,
                      num_gen_tokens=len(tokenizer.encode(record['response_text'], add_special_tokens=False)))
        recovered.append(record)
    if recovered:
        with (output / 'candidates.progress.jsonl').open('a', encoding='utf-8') as destination:
            destination.write(json.dumps({'records': recovered}, ensure_ascii=False) + '\n')
            destination.flush()
        for record in recovered:
            completed[record['sample_idx'], record['candidate_id']] = record
    pending_path = output / 'pending_reviews'
    pending_path.mkdir(exist_ok=True)
    for key, record in sorted(exhausted.items()):
        if key in completed or key in pending:
            continue
        base.write_json(pending_path / f'{key[0]:05d}_{key[1]}.json',
                        dict(record, status='pending_native_context_review', config=config))
        pending.add(key)
    return len(recovered)


def run(args):
    root = args.root.resolve()
    base.configure_environment(root)
    import transformers
    model_path = args.model_path.resolve()
    output = args.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = json.loads((root / 'data/processed' / args.dataset / (args.split + '.json')).read_text('utf-8'))
    ids_path = args.ids if args.ids.is_absolute() else root / args.ids
    request = request_from_ids(json.loads(ids_path.read_text('utf-8')), args.dataset)
    paper = json.loads((root / 'configs/paper_experiments.json').read_text('utf-8'))
    allowed = {c['generator_id']: c['generator_name'] for c in paper['centers']}
    if allowed.get(args.generator_id) != args.model_name:
        raise ValueError('Use one of the configured original frozen generator identities')
    indices = selected_indices(request, args.question_start, args.question_end, len(rows))
    config = dict(run_id=RUN, model_path=str(model_path), model_name=args.model_name,
        generator_id=args.generator_id, dataset=args.dataset, split=args.split, indices=indices,
        reference_ids=str(args.ids), num_candidates=8, seed=42, parser='final-answer-v3-2026-10-21',
        initial_max_new_tokens=4096, tensor_parallel=args.tp)
    path = output / 'config.json'
    if path.exists() and json.loads(path.read_text('utf-8')) != config:
        raise ValueError('Existing generation shard has different inputs')
    base.write_json(path, config)
    if (output / 'complete.json').exists():
        saved = json.loads((output / 'candidates.json').read_text('utf-8'))
        if [s['sample_idx'] for s in saved] != indices or any(len(s['candidates']) != 8 for s in saved):
            raise ValueError('Complete generation record has incomplete samples')
        return
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path, local_files_only=True)
    native = int(json.loads((model_path / 'config.json').read_text('utf-8'))['max_position_embeddings'])
    context = min(native, 32768)
    prompt_module = base.import_source(root, 'prompt_loader')
    extractor = base.import_source(root, 'answer_extractor')
    if extractor.ANSWER_PARSER_VERSION != config['parser']: raise ValueError('Wrong answer parser')
    prompts = {i: tokenizer.apply_chat_template([{'role': 'user',
        'content': prompt_module.load_prompts(args.dataset) + '\n\nQ: ' + rows[i]['question']}],
        tokenize=False, add_generation_prompt=True) for i in indices}
    lengths = {i: len(tokenizer.encode(p, add_special_tokens=True)) for i, p in prompts.items()}
    if max(lengths.values()) + 3 >= native: raise ValueError('Question exceeds native context')
    while max(lengths.values()) + 3 >= context: context = min(native, context * 2)
    settings = sampling_settings(root, args.dataset, args.split)
    journal = output / 'candidates.progress.jsonl'
    completed = base.load_progress(journal)
    expected = {(i, c) for i in indices for c in range(8)}
    if set(completed) - expected: raise ValueError('Journal has unrelated question IDs')
    pending_path = output / 'pending_reviews'
    pending_path.mkdir(exist_ok=True)
    pending = {(int(p.stem.split('_')[0]), int(p.stem.split('_')[1])) for p in pending_path.glob('*.json')}
    restore_public_captures(output, completed, pending, rows, indices, lengths,
                            settings, native, extractor, tokenizer, config)
    server = None
    started = time.monotonic()
    try:
        for batch_number in sorted({i // 128 for i in indices}):
            while True:
                missing = [(i, c) for i in indices if i // 128 == batch_number for c in range(8)
                           if (i, c) not in completed and (i, c) not in pending]
                if not missing: break
                if server is None:
                    server = HeterogeneousServer(root, model_path, args.model_name, args.port, context, args.tp, server_python=sys.executable)
                    server.log_path = output / 'server.log'
                    server.start()
                temperature, top_p = settings[batch_number]
                available = [context - lengths[i] - 2 for i, _ in missing]
                attempts = {}

                def capture(offset, response, budget):
                    index, cid = missing[offset]
                    record = dict(response, sample_idx=index, candidate_id=cid, temperature=temperature,
                                  top_p=top_p, max_new_tokens_used=budget, context_length=context)
                    attempts[offset] = record
                    with (output / 'response_attempts.jsonl').open('a', encoding='utf-8') as stream:
                        stream.write(json.dumps(record, ensure_ascii=False) + '\n'); stream.flush()
                    if finished(record):
                        answer = extractor.match_choice_answer(extractor.extract_answer(record['response_text']), rows[index]['options'])
                        reward = extractor.compute_reward(answer, rows[index])
                        record.update(extracted_answer=answer, reward=reward, is_correct=reward == 1.0,
                            num_gen_tokens=len(tokenizer.encode(record['response_text'], add_special_tokens=False)))
                        key = (index, cid)
                        if key in completed: return
                        with journal.open('a', encoding='utf-8') as stream:
                            stream.write(json.dumps({'records': [record]}, ensure_ascii=False) + '\n'); stream.flush()
                        completed[key] = record

                try:
                    base.complete_without_truncation(server.complete, [prompts[i] for i, _ in missing],
                        temperature, top_p, 4096, available, on_response=capture)
                except RuntimeError as error:
                    if 'available context limit' not in str(error): raise
                    if context < native:
                        server.close(); server = None; context = min(native, context * 2)
                        continue
                    for offset, record in attempts.items():
                        if record['finish_reason'] == 'length' and record['max_new_tokens_used'] >= available[offset]:
                            key = (record['sample_idx'], record['candidate_id'])
                            base.write_json(pending_path / f'{key[0]:05d}_{key[1]}.json',
                                dict(record, status='pending_native_context_review', config=config))
                            pending.add(key)
                    if not pending: raise
                print(json.dumps(dict(event='generation_progress', completed=len(completed),
                                      expected=len(expected), pending=len(pending))), flush=True)
        if pending:
            base.write_json(output / 'pending.json', dict(status='pending_native_context_review',
                completed=len(completed), expected=len(expected), pending=sorted(pending)))
            raise RuntimeError('Saved native-context attempts need review; no incomplete answer was scored')
        samples = assemble_selected(rows, indices, completed, args.dataset, args.split, args.generator_id)
        base.write_json(output / 'candidates.json', samples)
        base.write_json(output / 'complete.json', dict(status='complete', config=config,
            questions=len(samples), candidates=len(completed), elapsed_seconds=time.monotonic() - started))
    finally:
        if server is not None: server.close()


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path(__file__).resolve().parents[1])
    p.add_argument('--model-path', type=Path, required=True)
    p.add_argument('--model-name', required=True)
    p.add_argument('--generator-id', required=True)
    p.add_argument('--dataset', required=True, choices=base.DATASETS)
    p.set_defaults(split='train')
    p.add_argument('--output-dir', type=Path, required=True)
    p.add_argument('--ids', type=Path, default=Path('configs/public_reference_ids.json'))
    p.add_argument('--question-start', type=int, default=0)
    p.add_argument('--question-end', type=int)
    p.add_argument('--port', type=int, default=29500)
    p.add_argument('--tp', type=int, default=1)
    return p


if __name__ == '__main__': run(parser().parse_args())
