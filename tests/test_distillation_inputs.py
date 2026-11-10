"""Mechanical checks for the portable soft-label input adapter."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
TEMP_ROOT = ROOT / "cache" / "tmp"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)

class DistillationInputs(unittest.TestCase):
    def test_soft_targets_preserve_source_order_and_ignore_correctness(self):
        self.assertTrue((ROOT / "scripts/distill_verifier.py").exists(), "Portable distillation entry point missing")
        from distill_verifier import load_distillation_sources
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as folder:
            p=Path(folder)
            samples=[dict(sample_idx=19, mode="train", question="Q", candidates=[dict(candidate_id=c,response_text="R",is_correct=True) for c in range(8)])]
            (p/"pool.json").write_text(json.dumps(samples),encoding="utf-8")
            labels=[dict(sample_idx=19,candidate_id=c,teacher_probability=c/10) for c in range(8)]
            (p/"labels.jsonl").write_text("".join(json.dumps(x)+"\n" for x in labels),encoding="utf-8")
            (p/"sources.json").write_text(json.dumps([dict(generator="Qwen2.5-7B-Instruct",candidates="pool.json",teacher_labels="labels.jsonl")]),encoding="utf-8")
            rows,sources=load_distillation_sources(p/"sources.json")
            self.assertEqual([r["target"] for r in rows],[c/10 for c in range(8)])
            self.assertEqual([r["candidate_id"] for r in rows],list(range(8)))
            self.assertEqual(rows[0]["sample_idx"],19)
            self.assertEqual(len(sources),1)
            labels[2]["sample_idx"]=20
            (p/"labels.jsonl").write_text("".join(json.dumps(x)+"\n" for x in labels),encoding="utf-8")
            with self.assertRaisesRegex(ValueError,"identity"):
                load_distillation_sources(p/"sources.json")

if __name__=="__main__":unittest.main()
