"""CPU checks for agreement among the released table/figure snapshots."""
import json
from pathlib import Path
import unittest


ROOT = Path(__file__).resolve().parents[1]


class PaperResultConsistency(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tables = json.loads((ROOT / "results" / "paper_tables.json").read_text(encoding="utf-8"))["tables"]
        cls.scaling = json.loads((ROOT / "results" / "candidate_scaling.json").read_text(encoding="utf-8"))

    def test_table_sources_exist(self):
        for table in self.tables.values():
            source = ROOT / "results" / table["latex"]
            self.assertTrue(source.is_file())
            self.assertIn("\\label{" + table["label"] + "}", source.read_text(encoding="utf-8"))

    def test_shared_heterogeneous_teacher(self):
        main = {r["target_backbone"]: r for r in self.tables["main_results"]["records"] if r["method"] == "FedVP (Heterogeneous)"}
        for name, method in [("model_heterogeneous_comparison", "FedVP (Heterogeneous)"), ("verifier_distillation", "FedVP (Teacher pool)")]:
            for row in self.tables[name]["records"]:
                if row["method"] != method:
                    continue
                reference = main[row["target_backbone"]]
                self.assertEqual(row["accuracy_percent"], reference["accuracy_percent"])
                self.assertEqual(row["average_percent"], reference["average_percent"])

    def test_scaling_uses_reported_prefixes(self):
        self.assertEqual(self.scaling["candidate_budgets"], [1, 2, 4, 8])
        rows = self.tables["uniform_training_by_dataset"]["records"]
        methods = {
            "FedVP (Uniform)": "FedVP (Uniform) (N=8)",
            "FedVP (Heterogeneous)": "FedVP (Heterogeneous) (N=8)",
            "Self-Consistency": "Self-Consistency (N=8)",
            "Pretrained Verifier": "Pretrained single verifier (N=8)",
        }
        for target, series in self.scaling["series"].items():
            selected = {r["method"]: r for r in rows if r["target_backbone"] == target}
            cot = selected["CoT (N=1)"]["average_percent"]
            for curve, values in series.items():
                self.assertEqual(len(values), 4)
                self.assertAlmostEqual(values[0], cot, delta=0.0051)
                if curve in methods:
                    self.assertAlmostEqual(values[-1], selected[methods[curve]]["average_percent"], delta=0.0051)

    def test_client_mapping(self):
        clients = self.tables["client_configuration"]["records"]
        self.assertEqual([r["client"] for r in clients], list("ABCDE"))
        cross = self.tables["cross_generator_baselines"]["records"]
        self.assertEqual([r["pretrained_diagonal"]["client"] for r in cross], list("ABCDE"))
        self.assertEqual([r["configuration"] for r in self.tables["heterogeneous_leave_one_out"]["records"]], ["Full pool", *["w/o Client " + x for x in "ABCDE"]])


if __name__ == "__main__":
    unittest.main()
