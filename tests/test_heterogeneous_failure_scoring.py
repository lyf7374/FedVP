"""Mechanical scoring coverage for the five approved failed test candidates."""
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
TEMP_ROOT = ROOT / "cache" / "tmp"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
import heterogeneous_verifier as verifier
import score_heterogeneous_verifier as scoring


def rows():
    common = dict(center="medmcqa", source_index=0, sample_idx=1196,
                  question="Q", response_text="answer")
    return [dict(common, candidate_id=0, target=1.),
            dict(common, candidate_id=1, target=0., response_text="loop " * 9000,
                 generation_failure=True, selection_eligible=False,
                 failure_reason="repetitive_generation_no_final_answer",
                 failure_authorization="configs/heterogeneous_generation_failures_20261024.json"),
            dict(common, candidate_id=2, target=0.)]


class ApprovedFailureScoringTests(unittest.TestCase):
    def test_failed_attempt_has_null_scores_and_roundtrips_on_resume(self):
        candidate_rows = rows()
        failure = verifier.score_record(candidate_rows[1], None)
        self.assertIsNone(failure["logit"])
        self.assertIsNone(failure["sigmoid_score"])
        self.assertFalse(failure["selection_eligible"])
        self.assertEqual(failure["candidate_id"], 1)
        self.assertTrue(failure["generation_failure"])
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            path = Path(folder) / "scores.jsonl"
            path.write_text(json.dumps(failure, allow_nan=False) + "\n", encoding="utf-8")
            restored = verifier.read_saved_scores(path, candidate_rows)
            self.assertEqual(restored[verifier.row_key(candidate_rows[1])], failure)
            changed = dict(failure, logit=100., sigmoid_score=1.)
            path.write_text(json.dumps(changed) + "\n", encoding="utf-8")
            with self.assertRaises(ValueError):
                verifier.read_saved_scores(path, candidate_rows)
        with self.assertRaises(ValueError):
            verifier.score_record(candidate_rows[0], None)

    def test_excluded_attempt_cannot_win_even_if_given_a_high_score(self):
        self.assertTrue(hasattr(verifier, "select_predictions"))
        candidate_rows = rows()
        saved = {verifier.row_key(candidate_rows[0]): verifier.score_record(candidate_rows[0], -10.),
                 verifier.row_key(candidate_rows[1]): dict(candidate_rows[1], sigmoid_score=1.),
                 verifier.row_key(candidate_rows[2]): verifier.score_record(candidate_rows[2], -10.)}
        selected = verifier.select_predictions(candidate_rows, saved)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0]["candidate_id"], 0)
        self.assertTrue(selected[0]["is_correct"])

    def test_scoring_never_tokenizes_failed_attempt_and_preserves_question(self):
        import torch
        candidate_rows = rows()
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            root = Path(folder)
            checkpoint = root / "checkpoint"
            checkpoint.mkdir()
            (checkpoint / "manifest.json").write_text(json.dumps({"config": {
                "center": "bioasq", "model_id": "test-verifier", "verifier_kind": "standard",
                "model_path": str(root / "base")}}))
            args = SimpleNamespace(batch_size=2, candidates=root / "candidates.json", center="medmcqa",
                generator="Llama-3.2-3B-Instruct", split="test", question_start=0, question_end=1,
                checkpoint=checkpoint, model_path=root / "base", output_dir=root / "scores", device="cpu")
            model = Mock()
            model.config.max_position_embeddings = 64
            observed = []
            def collate(tokenizer, batch_rows, native_context, device):
                self.assertTrue(all(row.get("selection_eligible", True) for row in batch_rows))
                observed.extend(row["candidate_id"] for row in batch_rows)
                return {}
            with patch.object(scoring, "load_candidate_sources", return_value=(candidate_rows, [])) as load, \
                 patch.object(scoring, "load_verifier", return_value=model), \
                 patch.object(scoring, "collate_candidates", side_effect=collate), \
                 patch.object(scoring, "forward_logits", return_value=torch.tensor([-10., -11.])), \
                 patch("transformers.AutoTokenizer.from_pretrained", return_value=Mock()):
                result = scoring.run_scoring(args)
            self.assertTrue(load.call_args.kwargs["allow_generation_failures"])
            self.assertEqual(observed, [0, 2])
            self.assertEqual(result["candidate_count"], 3)
            self.assertEqual(result["questions"], 1)
            self.assertEqual(result["correct"], 1)
            records = [json.loads(line) for line in (root / "scores/candidate_scores.jsonl").read_text().splitlines()]
            self.assertEqual({record["candidate_id"] for record in records}, {0, 1, 2})
            self.assertIsNone(next(record for record in records if record["candidate_id"] == 1)["logit"])


if __name__ == "__main__":
    unittest.main()
