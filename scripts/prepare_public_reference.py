"""Materialize the saved public reference and assemble its existing candidates.

This is a CPU-only identity-preserving export. It never resamples questions or
generates responses. Candidate files may be complete original training pools or
selected-public pools from generate_public_candidates.py.
"""
import argparse
from collections import Counter
from copy import deepcopy
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def publish(path, value):
    path = Path(path)
    if path.exists():
        if read(path) != value:
            raise ValueError(f'Existing output differs: {path}')
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + '.writing')
    temporary.write_text(json.dumps(value, ensure_ascii=False, allow_nan=False,
                                    indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def materialize_rows(spec, training):
    records = spec['records']
    identities = [(r['dataset'], r['original_train_index']) for r in records]
    if len(identities) != spec['total_questions'] or len(set(identities)) != len(identities):
        raise ValueError('Reference IDs are incomplete or duplicated')
    if dict(Counter(c for c, _ in identities)) != spec['public_counts']:
        raise ValueError('Reference per-center counts differ')
    expected = [(c, i) for c in spec['center_order']
                for i in sorted(i for d, i in identities if d == c)]
    if expected != identities:
        raise ValueError('Reference IDs must preserve center/original-index order')
    public = []
    for center, index in identities:
        if not isinstance(index, int) or not 0 <= index < len(training[center]):
            raise ValueError('Reference index is outside its original train split')
        value = deepcopy(training[center][index])
        value.update(source_center=center, source_index=index, source_split='train',
                     uid=f'{center}:train:{index}', is_public=True)
        public.append(value)
    return public


def assemble_candidates(public, pools, generator_id):
    by_center = {}
    for center, rows in pools.items():
        by_index = {r['sample_idx']: r for r in rows}
        if len(by_index) != len(rows):
            raise ValueError(f'Duplicate original question indices in {center}')
        by_center[center] = by_index
    output = []
    for index, reference in enumerate(public):
        center, source_index = reference['source_center'], reference['source_index']
        sample = by_center.get(center, {}).get(source_index)
        if sample is None:
            raise ValueError(f'Missing candidate question {center}:{source_index}')
        if sample.get('mode') != 'train' or sample.get('dataset_name') != center:
            raise ValueError('Candidate input must retain its original center/train identity')
        for field in ('question', 'options', 'answer_string', 'answer_letter', 'answer_idx'):
            if sample.get(field) != reference.get(field):
                raise ValueError(f'Candidate QA metadata differs: {center}:{source_index}:{field}')
        recorded_generator = sample.get('source_generator_id', sample.get('generator_id'))
        if recorded_generator is not None and recorded_generator != generator_id:
            raise ValueError('Candidate source generator differs')
        candidates = sample['candidates']
        if [c['candidate_id'] for c in candidates] != list(range(8)):
            raise ValueError('Public training requires all eight ordered candidates')
        for candidate in candidates:
            if (candidate.get('finish_reason') != 'stop' or candidate.get('generation_failure')
                    or candidate.get('selection_eligible') is False
                    or not isinstance(candidate.get('response_text'), str)
                    or not candidate['response_text']):
                raise ValueError('Public training candidates must be complete saved responses')
        result = deepcopy(sample)
        result.update(dataset_name='public', mode='train', sample_idx=index,
                      source_center=center, source_index=source_index, source_split='train',
                      uid=reference['uid'], is_public=True, source_generator_id=generator_id,
                      parser_version='final-answer-v3-2026-10-21')
        output.append(result)
    return output


def load_public(root, ids):
    spec = read(ids)
    training = {c: read(root / 'data/processed' / c / 'train.json')
                for c in spec['center_order']}
    return materialize_rows(spec, training)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('action', choices=('materialize', 'assemble'))
    parser.add_argument('--root', type=Path, default=ROOT)
    parser.add_argument('--ids', type=Path, default=Path('configs/public_reference_ids.json'))
    parser.add_argument('--output', type=Path)
    parser.add_argument('--generator-id', help='Required by assemble, e.g. Qwen/Qwen2.5-7B-Instruct')
    parser.add_argument('--pool', action='append', default=[], metavar='DATASET=FILE',
                        help='Repeat for each source dataset; FILE may be a full or selected train pool')
    args = parser.parse_args()
    root = args.root.resolve()
    resolve = lambda path: path if path.is_absolute() else root / path
    public = load_public(root, resolve(args.ids))
    if args.action == 'materialize':
        output = resolve(args.output or Path('data/processed/public/train.json'))
        result = public
    else:
        if not args.generator_id or args.output is None or not args.pool:
            parser.error('assemble requires --generator-id, --output and --pool DATASET=FILE')
        pools = {}
        for item in args.pool:
            center, separator, path = item.partition('=')
            if not separator or center in pools:
                parser.error('Use one --pool DATASET=FILE per dataset')
            pools[center] = read(resolve(Path(path)))
        result = assemble_candidates(public, pools, args.generator_id)
        output = resolve(args.output)
    publish(output, result)
    print(json.dumps({'output': str(output), 'questions': len(result),
                      'candidates': sum(len(r.get('candidates', [])) for r in result)}))


if __name__ == '__main__':
    main()
