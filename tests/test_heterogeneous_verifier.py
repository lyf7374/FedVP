"""CPU-only mechanical checks for native-head training and resumable score IDs."""
import importlib
import json
import os
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
TEMP_ROOT = ROOT / "cache" / "tmp"
TEMP_ROOT.mkdir(parents=True, exist_ok=True)
os.environ["CUDA_VISIBLE_DEVICES"] = ""


def implementation():
    assert (ROOT / "scripts/heterogeneous_verifier.py").exists(), "Missing native-head implementation"
    return importlib.import_module("heterogeneous_verifier")


class MechanicalTests(unittest.TestCase):
    def test_question_range_and_resume_preserve_actual_ids(self):
        m = implementation()
        rows = [dict(center="medqa", source_index=0, sample_idx=s, candidate_id=c,
                     target=0., question="Q", response_text="A") for s in (19, 7, 30) for c in (4, 8)]
        selected = m.question_range(rows, 1, 3)
        self.assertEqual([(r["sample_idx"], r["candidate_id"]) for r in selected],
                         [(7, 4), (7, 8), (30, 4), (30, 8)])
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            path = Path(temp) / "scores.jsonl"
            records = [m.score_record(selected[1], .4), m.score_record(selected[0], -.2)]
            path.write_text("".join(json.dumps(r) + "\n" for r in records), encoding="utf-8")
            saved = m.read_saved_scores(path, selected)
            self.assertEqual(len(saved), 2)
            self.assertAlmostEqual(saved[m.row_key(selected[1])]["logit"], .4)
            path.write_text(json.dumps(records[0]), encoding="utf-8")
            m.read_saved_scores(path, selected)
            self.assertTrue(path.read_text(encoding="utf-8").endswith("\n"))
            path.write_text(json.dumps({**records[0], "sample_idx": 999}) + "\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "outside"):
                m.read_saved_scores(path, selected)

    def test_native_armorm_loader_preserves_all_heads_through_lora(self):
        m = implementation()
        import torch
        from transformers import LlamaConfig
        from train_heterogeneous_verifier import training_defaults, save_checkpoint
        torch.set_num_threads(1)
        torch.manual_seed(42)
        config = training_defaults()
        config.update(verifier_kind="armorm", model_id="tiny-armorm", rank=2, alpha=4, dropout=0.)
        native = m.armorm_model_class()(LlamaConfig(vocab_size=128020, hidden_size=16,
                    intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                    num_key_value_heads=1, max_position_embeddings=64, pad_token_id=0,
                    num_objectives=3, gating_hidden_dim=8, gating_n_hidden=2,
                    attn_implementation="sdpa"))
        ids = torch.tensor([[2, 128009, 128006, 78191, 128007, 271, 9, 128009]])
        batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            temp = Path(temp)
            native.save_pretrained(temp / "base")
            model = m.load_verifier(temp / "base", config, torch.device("cpu"))
            optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            loss = torch.nn.functional.binary_cross_entropy_with_logits(m.forward_logits(model, batch), torch.ones(1))
            loss.backward()
            self.assertGreater(model.regression_layer.weight.grad.abs().sum().item(), 0)
            self.assertGreater(model.gating.logit_scale.grad.abs().sum().item(), 0)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for n, p in model.named_parameters() if "lora_B" in n))
            optimizer.step()
            save_checkpoint(model, None, optimizer, temp / "checkpoint", config,
                            dict(epoch=0, next_batch=1, optimizer_steps=1), torch.device("cpu"))
            restored = m.load_verifier(temp / "base", config, torch.device("cpu"),
                                      checkpoint=temp / "checkpoint", training=False)
            model.eval()
            restored.eval()
            with torch.no_grad():
                torch.testing.assert_close(m.forward_logits(model, batch), m.forward_logits(restored, batch),
                                           rtol=0, atol=0)

    def test_native_armorm_math_gradients_and_head_roundtrip(self):
        m = implementation()
        import torch
        from torch import nn
        from types import SimpleNamespace
        torch.set_num_threads(1)
        torch.manual_seed(7)

        class Backbone(nn.Module):
            def __init__(self):
                super().__init__()
                self.embedding = nn.Embedding(128020, 6)
            def forward(self, input_ids, **kwargs):
                return SimpleNamespace(last_hidden_state=self.embedding(input_ids))

        class Native(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = Backbone()
                self.regression_layer = nn.Linear(6, 3, bias=False)
                self.gating = nn.Sequential(nn.Linear(6, 3), nn.Softmax(dim=-1))
                self.reward_transform_matrix = nn.Parameter(torch.eye(3), requires_grad=False)

        model = Native()
        model._verifier_kind = "armorm"
        ids = torch.tensor([[2, 128009, 128006, 78191, 128007, 271, 9, 128009]])
        batch = dict(input_ids=ids, attention_mask=torch.ones_like(ids))
        actual = m.forward_logits(model, batch)
        hidden = model.model(ids).last_hidden_state
        expected = ((model.regression_layer(hidden[:, -1]) @ model.reward_transform_matrix)
                    * model.gating(hidden[:, 1])).sum(-1)
        torch.testing.assert_close(actual, expected)
        torch.nn.functional.binary_cross_entropy_with_logits(actual, torch.ones(1)).backward()
        for part in (model.model.embedding, model.regression_layer, model.gating[0]):
            self.assertGreater(part.weight.grad.abs().sum().item(), 0)
        self.assertIsNone(model.reward_transform_matrix.grad)
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            m.save_native_head(model, Path(temp))
            replacement = Native()
            replacement._verifier_kind = "armorm"
            replacement.model.load_state_dict(model.model.state_dict())
            m.restore_native_head(replacement, Path(temp))
            torch.testing.assert_close(m.forward_logits(replacement, batch), actual.detach(), rtol=0, atol=0)

    def test_standard_lora_checkpoint_roundtrip_and_accumulation(self):
        m = implementation()
        import copy
        import torch
        from transformers import LlamaConfig, LlamaForSequenceClassification
        from train_skywork_verifier import training_defaults, collate_candidates
        from train_heterogeneous_verifier import train_effective_batch, save_checkpoint
        torch.set_num_threads(1)
        torch.manual_seed(42)
        config = training_defaults()
        config.update(verifier_kind="standard", model_id="tiny", rank=2, alpha=4, dropout=0.)
        base = LlamaForSequenceClassification(LlamaConfig(vocab_size=32, hidden_size=16,
                    intermediate_size=32, num_hidden_layers=1, num_attention_heads=2,
                    num_key_value_heads=1, num_labels=1, max_position_embeddings=64,
                    pad_token_id=0, attn_implementation="sdpa"))
        class Tokenizer:
            pad_token_id = 0
            def apply_chat_template(self, messages, **kwargs):
                return [1, 3, 2] if messages[-1]["content"] == "short" else [1, 4, 5, 2]
        tokenizer = Tokenizer()
        with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp:
            temp = Path(temp)
            base.save_pretrained(temp / "base")
            model = m.prepare_lora(base, config, torch.device("cpu"))
            copy_model = copy.deepcopy(model)
            rows = [dict(question="Q", response_text=text, target=target)
                    for text, target in [("short", 1.), ("long", 0.), ("short", 0.)]]
            opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-4)
            opt2 = torch.optim.AdamW([p for p in copy_model.parameters() if p.requires_grad], lr=1e-4)
            a = train_effective_batch(model, tokenizer, rows, opt, torch.device("cpu"), 1, 64)
            b = train_effective_batch(copy_model, tokenizer, rows, opt2, torch.device("cpu"), 3, 64)
            self.assertAlmostEqual(a, b, places=6)
            self.assertGreater(model.score.weight.grad.abs().sum().item(), 0)
            self.assertTrue(any(p.grad is not None and p.grad.abs().sum() > 0
                                for n, p in model.named_parameters() if "lora_B" in n))
            save_checkpoint(model, None, opt, temp / "checkpoint", config,
                            dict(epoch=0, next_batch=1, optimizer_steps=1), torch.device("cpu"))
            restored = m.load_verifier(temp / "base", config, torch.device("cpu"),
                                      checkpoint=temp / "checkpoint", training=False)
            model.eval()
            restored.eval()
            batch = collate_candidates(tokenizer, rows, 64, torch.device("cpu"))
            with torch.no_grad():
                torch.testing.assert_close(m.forward_logits(model, batch), m.forward_logits(restored, batch),
                                           rtol=0, atol=0)


if __name__ == "__main__":
    unittest.main()
