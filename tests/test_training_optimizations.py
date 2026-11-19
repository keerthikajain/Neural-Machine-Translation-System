"""Focused regression checks for Saarthi training optimizations.

Run: python -m unittest discover -s tests -v
"""
import os
import tempfile
import unittest

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.trainer import LabelSmoothingCrossEntropy, NMTTrainer


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.embedding = nn.Module()
        self.encoder.embedding.token_embedding = nn.Module()
        self.encoder.embedding.token_embedding.d_model = 8
        self.embedding = nn.Embedding(16, 8)
        self.output = nn.Linear(8, 16)

    def forward(self, src_tokens, tgt_tokens):
        return self.output(self.embedding(tgt_tokens))


def make_batch():
    return {
        "src_tokens": torch.tensor([[2, 4, 0], [2, 5, 6]]),
        "tgt_input": torch.tensor([[2, 7, 0], [2, 8, 9]]),
        "tgt_output": torch.tensor([[7, 3, 0], [8, 9, 3]]),
    }


class TrainingOptimizationTests(unittest.TestCase):
    def test_label_smoothing_matches_previous_formula_and_gradients(self):
        torch.manual_seed(7)
        logits = torch.randn(8, 16, dtype=torch.float32, requires_grad=True)
        labels = torch.tensor([1, 2, 0, 4, 5, 0, 8, 3])
        smoothing = 0.1
        mask = labels != 0
        log_probs = torch.log_softmax(logits[mask], dim=-1)
        targets = labels[mask]
        distribution = torch.full_like(log_probs, smoothing / 15)
        distribution.scatter_(1, targets[:, None], 1 - smoothing)
        baseline = -(distribution * log_probs).sum(-1).mean()
        baseline_grad = torch.autograd.grad(baseline, logits, retain_graph=True)[0]
        optimized = LabelSmoothingCrossEntropy(16, smoothing, 0)(logits, labels)
        optimized_grad = torch.autograd.grad(optimized, logits)[0]
        torch.testing.assert_close(optimized, baseline, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(optimized_grad, baseline_grad, atol=1e-6, rtol=1e-6)

    def test_cpu_step_and_checkpoint_resume_epoch(self):
        with tempfile.TemporaryDirectory() as folder:
            batch = make_batch()
            loader = DataLoader([batch], batch_size=None)
            trainer = NMTTrainer(
                model=TinyModel(), train_loader=loader, val_loader=loader,
                device=torch.device("cpu"), vocab_size=16,
                checkpoint_dir=folder, warmup_steps=5,
            )
            self.assertFalse(trainer.use_amp)
            self.assertGreater(trainer.train_step(batch), 0)
            trainer.epoch = 2
            path = os.path.join(folder, "epoch.pt")
            trainer.save_checkpoint(path)
            other = NMTTrainer(
                model=TinyModel(), train_loader=loader, val_loader=loader,
                device=torch.device("cpu"), vocab_size=16,
                checkpoint_dir=folder, warmup_steps=5,
            )
            other.load_checkpoint(path)
            self.assertEqual(other.epoch, 3)
            self.assertEqual(other.step, trainer.step)
            # A checkpoint written partway through an epoch repeats that epoch.
            trainer.save_checkpoint(path, epoch_complete=False)
            other.load_checkpoint(path)
            self.assertEqual(other.epoch, 2)


if __name__ == "__main__":
    unittest.main()
