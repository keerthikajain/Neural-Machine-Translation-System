import os
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader, Dataset

from src.model import (
    COMPACT_A_CONFIG,
    create_compact_a_model,
    create_transformer_model,
)
from src.model.decoder import create_causal_mask, create_target_mask
from src.training.trainer import NMTTrainer


class Phase3CompactATests(unittest.TestCase):
    def test_compact_a_configuration(self):
        self.assertEqual(COMPACT_A_CONFIG['d_model'], 256)
        self.assertEqual(COMPACT_A_CONFIG['n_heads'], 4)
        self.assertEqual(COMPACT_A_CONFIG['encoder_layers'], 4)
        self.assertEqual(COMPACT_A_CONFIG['decoder_layers'], 4)
        self.assertEqual(COMPACT_A_CONFIG['d_ff'], 1024)
        self.assertEqual(COMPACT_A_CONFIG['max_seq_length'], 128)
        self.assertEqual(COMPACT_A_CONFIG['vocab_size'], 32000)

    def test_compact_a_parameter_count(self):
        model = create_compact_a_model(vocab_size=32000)
        info = model.get_model_info()
        self.assertEqual(info['total_parameters'], 31980800)
        self.assertEqual(info['d_model'], 256)
        self.assertEqual(info['n_heads'], 4)
        self.assertEqual(info['d_ff'], 1024)
        self.assertEqual(info['max_seq_length'], 128)
        self.assertEqual(info['encoder_layers'], 4)
        self.assertEqual(info['decoder_layers'], 4)

    def test_reference_parameter_count_is_preserved(self):
        reference = create_transformer_model(vocab_size=32000)
        self.assertEqual(reference.get_model_info()['total_parameters'], 93322496)

    def test_compact_a_forward_shapes(self):
        model = create_compact_a_model(vocab_size=32000)
        model.eval()
        src = torch.randint(1, 32000, (2, 32))
        tgt = torch.randint(1, 32000, (2, 24))
        with torch.no_grad():
            logits = model(src, tgt)
        self.assertEqual(tuple(logits.shape), (2, 24, 32000))
        self.assertTrue(torch.isfinite(logits).all())

    def test_compact_a_padding_and_gradient_flow(self):
        model = create_compact_a_model(vocab_size=32000)
        model.train()
        src = torch.tensor([[2, 10, 11, 0, 0], [2, 12, 13, 14, 0]])
        tgt = torch.tensor([[2, 20, 21, 0, 0], [2, 22, 23, 24, 0]])
        logits = model(src, tgt)
        loss = logits[:, :4].sum()
        loss.backward()
        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(any(p.grad is not None and torch.isfinite(p.grad).all()
                            for p in model.parameters()))
        self.assertTrue(torch.equal(logits[:, 4:], torch.zeros_like(logits[:, 4:])))

    def test_compact_a_masks(self):
        tgt = torch.tensor([[2, 20, 21, 0, 0]])
        causal = create_causal_mask(5, torch.device('cpu'))
        target_mask = create_target_mask(tgt, pad_token_id=0)
        self.assertEqual(tuple(causal.shape), (1, 1, 5, 5))
        self.assertEqual(tuple(target_mask.shape), (1, 1, 5, 5))
        self.assertTrue(torch.all(target_mask[0, 0, 3:] == 0))
        self.assertTrue(torch.all(target_mask[0, 0, :, 3:] == 0))


class TinyDataset(Dataset):
    def __len__(self):
        return 2

    def __getitem__(self, index):
        return {
            'src_tokens': torch.tensor([2, 10, 11]),
            'tgt_input': torch.tensor([2, 20, 21]),
            'tgt_output': torch.tensor([20, 21, 3]),
        }


def tiny_collate(batch):
    return {
        'src_tokens': torch.stack([item['src_tokens'] for item in batch]),
        'tgt_input': torch.stack([item['tgt_input'] for item in batch]),
        'tgt_output': torch.stack([item['tgt_output'] for item in batch]),
    }


class Phase3CheckpointTests(unittest.TestCase):
    def test_compact_a_checkpoint_save_load(self):
        model = create_compact_a_model(vocab_size=32000)
        loader = DataLoader(TinyDataset(), batch_size=2, collate_fn=tiny_collate)

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            trainer = NMTTrainer(
                model=model,
                train_loader=loader,
                val_loader=loader,
                device=torch.device('cpu'),
                learning_rate=0.0001,
                warmup_steps=4,
                vocab_size=32000,
                checkpoint_dir=checkpoint_dir,
            )
            trainer.epoch = 2
            trainer.step = 17
            trainer.scheduler.step_num = 17
            checkpoint_path = os.path.join(checkpoint_dir, 'compact_a.pt')
            trainer.save_checkpoint(checkpoint_path)

            reloaded_model = create_compact_a_model(vocab_size=32000)
            reloaded = NMTTrainer(
                model=reloaded_model,
                train_loader=loader,
                val_loader=loader,
                device=torch.device('cpu'),
                learning_rate=0.0001,
                warmup_steps=4,
                vocab_size=32000,
                checkpoint_dir=checkpoint_dir,
            )
            reloaded.load_checkpoint(checkpoint_path)
            saved = torch.load(checkpoint_path, map_location='cpu')

            self.assertEqual(reloaded.epoch, 2)
            self.assertEqual(reloaded.step, 17)
            self.assertEqual(reloaded.scheduler.step_num, 17)
            self.assertEqual(reloaded._resume_epoch, 3)
            self.assertEqual(saved['architecture']['d_model'], 256)
            self.assertEqual(saved['architecture']['encoder_layers'], 4)
            self.assertEqual(saved['architecture']['decoder_layers'], 4)
            self.assertEqual(saved['architecture']['total_parameters'], 31980800)


if __name__ == '__main__':
    unittest.main()
