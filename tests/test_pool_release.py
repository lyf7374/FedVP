import importlib.util
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

class PoolTests(unittest.TestCase):
    def test_evaluation_module_exists(self):
        self.assertIsNotNone(importlib.util.find_spec("evaluate_pool"))

    def test_sigmoid_saturation_uses_first_candidate(self):
        from evaluate_pool import evaluate
        samples=[{"sample_idx":0,"candidates":[
            {"candidate_id":0,"is_correct":True,"extracted_answer":"A"},
            {"candidate_id":1,"is_correct":False,"extracted_answer":"B"}]}]
        result=evaluate(samples,{"A":{(0,0):100.0,(0,1):1000.0}},n=2,expected_questions=1)
        self.assertEqual(result["correct"],1)
        self.assertEqual(result["predictions"][0]["candidate_id"],0)

    def test_pool_and_removed_member(self):
        from evaluate_pool import evaluate
        samples=[{"sample_idx":0,"candidates":[
            {"candidate_id":0,"is_correct":True,"extracted_answer":"A"},
            {"candidate_id":1,"is_correct":False,"extracted_answer":"B"}]}]
        scores={"A":{(0,0):3.,(0,1):-3.},"B":{(0,0):-1.,(0,1):1.}}
        self.assertEqual(evaluate(samples,scores,n=2,expected_questions=1)["correct"],1)
        self.assertEqual(evaluate(samples,scores,n=2,expected_questions=1,drop_member="A")["correct"],0)

    def test_missing_score_is_not_complete(self):
        from evaluate_pool import evaluate
        samples=[{"sample_idx":0,"candidates":[{"candidate_id":0,"is_correct":True},{"candidate_id":1,"is_correct":False}]}]
        with self.assertRaises(ValueError):
            evaluate(samples,{"A":{(0,0):3.}},n=2,expected_questions=1)

class MacroTests(unittest.TestCase):
    def test_macro_weights_datasets_not_questions(self):
        import json
        import tempfile
        from summarize_accuracy import summarize,COUNTS
        scratch=Path(__file__).resolve().parents[1]/"cache"/"tmp"
        scratch.mkdir(parents=True,exist_ok=True)
        with tempfile.TemporaryDirectory(dir=scratch) as directory:
            paths={}
            for dataset,count in COUNTS.items():
                path=Path(directory)/f"{dataset}.json"
                path.write_text(json.dumps({"status":"complete","questions":count,
                    "correct":count if dataset=="bioasq" else 0}),encoding="utf-8")
                paths[dataset]=path
            self.assertEqual(summarize(paths)["mean_accuracy_percent"],20.0)
            with self.assertRaises(ValueError):
                summarize({k:v for k,v in paths.items() if k!="medqa"})

if __name__=="__main__": unittest.main()
