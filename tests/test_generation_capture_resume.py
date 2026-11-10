"""CPU checks that interrupted generation never repeats captured completions."""
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest import mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import run_candidates as generation


class Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return text.split()

    def apply_chat_template(self, messages, tokenize=False, add_generation_prompt=True):
        return "question"


class Extractor:
    def extract_answer(self, text):
        return text

    def match_choice_answer(self, answer, options):
        return answer if answer in options else None

    def compute_reward(self, answer, row):
        return float(answer == row["answer_letter"])


class CaptureRecovery(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.path = Path(self.temp.name)
        self.completed = {}
        self.rows = [{"question": "question", "options": {"A": "one", "B": "two"}, "answer_letter": "A"}]

    def tearDown(self):
        self.temp.cleanup()

    def attempt(self, candidate, text="A", reason="stop", context=32, budget=8):
        return dict(sample_idx=0, candidate_id=candidate, response_text=text,
                    finish_reason=reason, dataset_name="bioasq", split="test",
                    original_batch=0, max_new_tokens=budget, available_context=context-2,
                    context_length=context, temperature=0.5, top_p=0.8, prompt="question")

    def save(self, *attempts):
        (self.path / "response_attempts.jsonl").write_text("".join(json.dumps(x)+"\n" for x in attempts), encoding="utf-8")

    def restore(self, context=32):
        return generation.restore_captured_responses(
            self.path, self.path / "candidates.progress.jsonl", self.completed,
            self.rows, ["question"], Extractor(), Tokenizer(), "bioasq", "test",
            [(0.5, 0.8)], context, 0, 1)

    def test_stop_restored_before_native_pending_refusal(self):
        self.save(self.attempt(0), self.attempt(1, "loop", "length", budget=30))
        with self.assertRaisesRegex(RuntimeError, "available context limit"):
            self.restore()
        self.assertEqual(self.completed[(0, 0)]["response_text"], "A")
        self.assertTrue(self.completed[(0, 0)]["is_correct"])
        self.assertNotIn((0, 1), self.completed)
        journal = self.path / "candidates.progress.jsonl"
        before = journal.read_bytes()
        with self.assertRaises(RuntimeError):
            self.restore()
        self.assertEqual(journal.read_bytes(), before)

    def test_smaller_execution_context_may_expand(self):
        self.save(self.attempt(0), self.attempt(1, "partial", "length", budget=30))
        self.assertEqual(self.restore(context=64), 1)
        self.assertEqual(set(self.completed), {(0, 0)})

    def test_ordinary_budget_truncation_does_not_block_retry(self):
        self.save(self.attempt(0, "partial", "length", budget=8))
        self.assertEqual(self.restore(), 0)
        self.assertEqual(self.completed, {})

    def test_later_stop_supersedes_earlier_truncation(self):
        self.save(self.attempt(0, "partial", "length", budget=30),
                  self.attempt(0, "B", "stop", context=64, budget=32))
        self.assertEqual(self.restore(context=64), 1)
        self.assertEqual(self.completed[(0, 0)]["response_text"], "B")
        self.assertFalse(self.completed[(0, 0)]["is_correct"])

    def test_existing_journal_record_is_never_replaced(self):
        self.completed[(0, 0)] = {"response_text": "original"}
        self.save(self.attempt(0, "B"))
        self.assertEqual(self.restore(), 0)
        self.assertEqual(self.completed[(0, 0)]["response_text"], "original")

    def test_capture_input_identity_must_match(self):
        wrong = self.attempt(0)
        wrong["prompt"] = "different question"
        self.save(wrong)
        with self.assertRaisesRegex(ValueError, "Captured response"):
            self.restore()
        self.assertFalse((self.path / "candidates.progress.jsonl").exists())

    def test_run_split_reuses_whole_captured_batch_without_inference(self):
        output = self.path / "target" / "bioasq_best_of_n" / "test"
        output.mkdir(parents=True)
        self.save(*[self.attempt(i) for i in range(8)])
        (output / "response_attempts.jsonl").write_bytes((self.path / "response_attempts.jsonl").read_bytes())
        server = mock.Mock()
        server.complete.side_effect = AssertionError("Captured stops must never be regenerated")
        prompt_module = mock.Mock()
        prompt_module.load_prompts.return_value = "prefix"
        with mock.patch.object(generation, "merge_candidate_shards") as merge:
            generation.run_split(self.path, self.path / "target", "bioasq", "test",
                                 self.rows, Tokenizer(), 32, server, prompt_module,
                                 Extractor(), [(0.5, 0.8)])
        server.complete.assert_not_called()
        merge.assert_called_once()
        self.assertEqual(len(generation.collect_candidate_progress(output)), 8)


if __name__ == "__main__":
    unittest.main()
