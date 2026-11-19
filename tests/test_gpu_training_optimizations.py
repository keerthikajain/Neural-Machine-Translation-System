"""Regression checks for opt-in FP16 and memory-efficient label smoothing."""
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from src.training.trainer import NMTTrainer, LabelSmoothingCrossEntropy


class TinyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = nn.Module()
        self.encoder.embedding = nn.Module()
        self.encoder.embedding.token_embedding = nn.Module()
        self.encoder.embedding.token_embedding.d_model = 8
        self.embed = nn.Embedding(16, 8)
        self.proj = nn.Linear(8, 16)

    def forward(self, src_tokens, tgt_tokens):
        return self.proj(self.embed(tgt_tokens))


def batch():
    return {
        "src_tokens": torch.tensor([[2, 4, 0], [2, 5, 6]]),
        "tgt_input": torch.tensor([[2, 7, 0], [2, 8, 9]]),
        "tgt_output": torch.tensor([[7, 3, 0], [8, 9, 3]]),
    }


class OptimizationsTest(unittest.TestCase):
    def test_loss_and_gradients_equal_original(self):
        torch.manual_seed(7)
        x = torch.randn(10, 16, requires_grad=True)
        y = torch.tensor([1, 2, 0, 4, 5, 0, 8, 3, 2, 1])
        valid = y != 0
        log_probs = torch.log_softmax(x[valid], dim=-1)
        dist = torch.full_like(log_probs, 0.1 / 15)
        dist.scatter_(1, y[valid, None], 0.9)
        old = -(dist * log_probs).sum(dim=-1).mean()
        old_grad = torch.autograd.grad(old, x, retain_graph=True)[0]

        new = LabelSmoothingCrossEntropy(16, 0.1, ignore_index=0)(x, y)
        new_grad = torch.autograd.grad(new, x)[0]
        torch.testing.assert_close(new, old, atol=1e-6, rtol=1e-6)
        torch.testing.assert_close(new_grad, old_grad, atol=1e-6, rtol=1e-6)

    def test_cpu_fallback_and_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            b = batch()
            loader = DataLoader([b], batch_size=None)
            trainer = NMTTrainer(
                TinyModel(), loader, loader, torch.device("cpu"),
                vocab_size=16, checkpoint_dir=root, use_amp=True
            )
            self.assertFalse(trainer.use_amp)
            self.assertGreater(trainer.train_step(b), 0)
            checkpoint = str(Path(root) / "test.pt")
            trainer.save_checkpoint(checkpoint)
            newer = NMTTrainer(
                TinyModel(), loader, loader, torch.device("cpu"),
                vocab_size=16, checkpoint_dir=root, use_amp=True
            )
            newer.load_checkpoint(checkpoint)
            self.assertEqual(newer.step, trainer.step)
            self.assertIn("amp_scaler_state_dict", torch.load(
                checkpoint, map_location="cpu", weights_only=False
            ))

    @unittest.skipUnless(torch.cuda.is_available(), "CUDA unavailable")
    def test_cuda_mixed_precision_step(self):
        with tempfile.TemporaryDirectory() as root:
            b = batch()
            loader = DataLoader([b], batch_size=None)
            trainer = NMTTrainer(
                TinyModel(), loader, loader, torch.device("cuda"),
                vocab_size=16, checkpoint_dir=root, use_amp=True
            )
            self.assertTrue(trainer.use_amp)
            loss = trainer.train_step(b)
            self.assertTrue(torch.isfinite(torch.tensor(loss)).item())


if __name__ == "__main__":
    unittest.main()
