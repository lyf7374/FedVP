import copy
import json
import sys
import tempfile
from pathlib import Path
import unittest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
from prepare_public_reference import materialize_rows, assemble_candidates
from generate_public_candidates import selected_indices, request_from_ids, restore_public_captures


def row(question):
    return dict(question=question, answer='#### A', answer_string='yes',
                answer_letter='A', answer_idx=0, options=['yes', 'no'], options_str='A. yes B. no')


class PublicReferenceTest(unittest.TestCase):
    def setUp(self):
        self.train = {'a': [row('a0'), row('a1')], 'b': [row('b0')]}
        self.ids = {'center_order': ['a', 'b'], 'public_counts': {'a': 1, 'b': 1},
                    'total_questions': 2, 'records': [
                        {'dataset': 'a', 'original_train_index': 1},
                        {'dataset': 'b', 'original_train_index': 0}]}
        self.public = materialize_rows(self.ids, self.train)
        self.pools = {}
        for center, index in [('a', 1), ('b', 0)]:
            sample = copy.deepcopy(self.train[center][index])
            sample.update(sample_idx=index, mode='train', dataset_name=center,
                          candidates=[dict(candidate_id=i, response_text=f'answer {i}',
                                           finish_reason='stop', extracted_answer='A',
                                           reward=1.0, is_correct=True) for i in range(8)])
            self.pools[center] = [sample]

    def test_preserves_selected_original_ids_and_source_order(self):
        self.assertEqual([x['uid'] for x in self.public], ['a:train:1', 'b:train:0'])
        self.assertEqual([x['question'] for x in self.public], ['a1', 'b0'])
        self.assertNotIn('uid', self.train['a'][1])

    def test_assemble_reindexes_only_public_union_and_preserves_candidates(self):
        result = assemble_candidates(self.public, self.pools, 'org/model')
        self.assertEqual([r['sample_idx'] for r in result], [0, 1])
        self.assertEqual([r['source_index'] for r in result], [1, 0])
        self.assertEqual(result[0]['candidates'], self.pools['a'][0]['candidates'])
        self.assertEqual(result[1]['dataset_name'], 'public')

    def test_incomplete_pool_and_qa_mismatch_are_rejected(self):
        bad = copy.deepcopy(self.pools)
        bad['b'][0]['candidates'].pop()
        with self.assertRaises(ValueError):
            assemble_candidates(self.public, bad, 'org/model')
        bad = copy.deepcopy(self.pools)
        bad['a'][0]['question'] = 'different question'
        with self.assertRaises(ValueError):
            assemble_candidates(self.public, bad, 'org/model')

    def test_length_limited_training_candidate_is_not_silently_accepted(self):
        bad = copy.deepcopy(self.pools)
        bad['a'][0]['candidates'][2]['finish_reason'] = 'length'
        with self.assertRaises(ValueError):
            assemble_candidates(self.public, bad, 'org/model')

    def test_public_generation_selects_only_saved_original_indices(self):
        request = request_from_ids(self.ids, 'a')
        self.assertEqual(selected_indices(request, 0, None, 2), [1])
        with self.assertRaises(ValueError):
            selected_indices(request, 0, None, 1)

    def test_capture_only_crash_restores_stop_and_native_pending_once(self):
        class Extractor:
            extract_answer = staticmethod(lambda text: text)
            match_choice_answer = staticmethod(lambda text, options: 'A')
            compute_reward = staticmethod(lambda answer, sample: 1.0)
        class Tokenizer:
            def encode(self, text, add_special_tokens=False):
                return text.split()
        with tempfile.TemporaryDirectory() as folder:
            output = Path(folder)
            common = dict(sample_idx=1, temperature=0.5, top_p=0.9, context_length=100)
            attempts = [dict(common, candidate_id=0, response_text='finished answer',
                             finish_reason='stop', max_new_tokens_used=88),
                        dict(common, candidate_id=1, response_text='unfinished loop',
                             finish_reason='length', max_new_tokens_used=88),
                        dict(common, candidate_id=2, response_text='ordinary truncated',
                             finish_reason='length', max_new_tokens_used=20)]
            path = output / 'response_attempts.jsonl'
            path.write_text(''.join(json.dumps(x)+'\n' for x in attempts), encoding='utf-8')
            before = path.read_bytes()
            completed, pending = {}, set()
            args = (output, completed, pending, self.train['a'], [1], {1:10},
                    [(0.5,0.9)], 100, Extractor(), Tokenizer(), {'dataset':'a'})
            self.assertEqual(restore_public_captures(*args), 1)
            self.assertEqual(set(completed), {(1,0)})
            self.assertEqual(pending, {(1,1)})
            saved = json.loads((output/'pending_reviews/00001_1.json').read_text())
            self.assertEqual(saved['finish_reason'], 'length')
            self.assertNotIn('generation_failure', saved)
            journal = (output/'candidates.progress.jsonl').read_bytes()
            self.assertEqual(restore_public_captures(*args), 0)
            self.assertEqual((output/'candidates.progress.jsonl').read_bytes(), journal)
            self.assertEqual(path.read_bytes(), before)


if __name__ == '__main__':
    unittest.main()
