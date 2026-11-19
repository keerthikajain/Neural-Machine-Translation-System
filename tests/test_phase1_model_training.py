import os
import tempfile
import unittest

import torch
from torch.utils.data import DataLoader

from src.model.attention import MultiHeadAttention
from src.model.decoder import create_causal_mask, create_target_mask
from src.model.transformer import create_transformer_model
from src.training.trainer import NMTTrainer


class PhaseOneMaskTests(unittest.TestCase):
    def test_causal_mask_masks_future_tokens(self):
        mask = create_causal_mask(4, torch.device('cpu'))
        matrix = mask[0, 0]
        self.assertEqual(tuple(mask.shape), (1, 1, 4, 4))
        self.assertTrue(torch.all(matrix == torch.tril(torch.ones(4, 4))))

    def test_target_padding_masks_keys_and_queries(self):
        tokens = torch.tensor([[2, 5, 3, 0], [2, 0, 0, 0]])
        mask = create_target_mask(tokens, pad_token_id=0)
        self.assertEqual(tuple(mask.shape), (2, 1, 4, 4))
        self.assertTrue(torch.all(mask[0, 0, 3] == 0))
        self.assertTrue(torch.all(mask[0, 0, :, 3] == 0))
        self.assertTrue(torch.all(mask[1, 0, 1:] == 0))

    def test_combined_causal_and_padding_mask(self):
        tokens = torch.tensor([[2, 5, 0, 0]])
        mask = create_target_mask(tokens, pad_token_id=0)
        matrix = mask[0, 0]
        self.assertEqual(matrix[0].tolist(), [1.0, 0.0, 0.0, 0.0])
        self.assertEqual(matrix[1].tolist(), [1.0, 1.0, 0.0, 0.0])
        self.assertTrue(torch.all(matrix[2:] == 0))

    def test_decoder_forward_with_padded_targets(self):
        model = create_transformer_model(vocab_size=32000, d_model=64, n_heads=4, n_layers=2)
        model.eval()
        src = torch.tensor([[2, 10, 11, 12, 0]])
        tgt = torch.tensor([[2, 20, 21, 0, 0]])
        mask = create_target_mask(tgt, pad_token_id=0)
        src_mask = (src != 0).unsqueeze(1).unsqueeze(1).float()
        with torch.no_grad():
            logits = model.decoder(tgt, model.encoder(src, src_mask), mask, src_mask)
        self.assertEqual(tuple(logits.shape), (1, 5, 32000))
        self.assertTrue(torch.isfinite(logits).all())


class PhaseOneCheckpointTests(unittest.TestCase):
    def test_checkpoint_save_reload_optimizer_scheduler_and_epoch(self):
        model = create_transformer_model(vocab_size=64, d_model=32, n_heads=4, n_layers=1)
        sample = torch.utils.data.Subset(
            torch.utils.data.TensorDataset(
                torch.randint(0, 64, (2, 5)),
                torch.randint(0, 64, (2, 5)),
                torch.randint(0, 64, (2, 5)),
            ),
            [0, 1],
        )
        # Build the trainer against a minimal DataLoader-compatible batch format.
        rows = []
        for src, tgt_in, tgt_out in sample:
            rows.append({
                'src_tokens': src,
                'tgt_input': tgt_in,
                'tgt_output': tgt_out,
            })
        from torch.utils.data import Dataset
        class TinyDataset(Dataset):
            def __len__(self): return len(rows)
            def __getitem__(self, index): return rows[index]
        def tiny_collate(batch):
            return {
                'src_tokens': torch.stack([x['src_tokens'] for x in batch]),
                'tgt_input': torch.stack([x['tgt_input'] for x in batch]),
                'tgt_output': torch.stack([x['tgt_output'] for x in batch]),
            }
        loader = DataLoader(TinyDataset(), batch_size=2, collate_fn=tiny_collate)

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            trainer = NMTTrainer(
                model=model,
                train_loader=loader,
                val_loader=loader,
                device=torch.device('cpu'),
                learning_rate=0.0001,
                warmup_steps=4,
                vocab_size=64,
                checkpoint_dir=checkpoint_dir,
            )
            trainer.epoch = 2
            trainer.step = 17
            trainer.scheduler.step_num = 17
            trainer.train_losses = [1.0]
            trainer.val_losses = [0.9]
            trainer.best_val_loss = 0.9

            checkpoint_path = os.path.join(checkpoint_dir, 'checkpoint_test.pt')
            trainer.save_checkpoint(checkpoint_path)

            reloaded_model = create_transformer_model(vocab_size=64, d_model=32, n_heads=4, n_layers=1)
            reloaded = NMTTrainer(
                model=reloaded_model,
                train_loader=loader,
                val_loader=loader,
                device=torch.device('cpu'),
                learning_rate=0.0001,
                warmup_steps=4,
                vocab_size=64,
                checkpoint_dir=checkpoint_dir,
            )
            reloaded.load_checkpoint(checkpoint_path)

            self.assertEqual(reloaded.epoch, 2)
            self.assertEqual(reloaded._resume_epoch, 3)
            self.assertEqual(reloaded.step, 17)
            self.assertEqual(reloaded.scheduler.step_num, 17)
            self.assertEqual(reloaded.best_val_loss, 0.9)
            for original, restored in zip(trainer.model.state_dict().values(), reloaded.model.state_dict().values()):
                self.assertTrue(torch.equal(original, restored))

            original_optimizer = trainer.optimizer.state_dict()['param_groups'][0]['lr']
            restored_optimizer = reloaded.optimizer.state_dict()['param_groups'][0]['lr']
            self.assertEqual(original_optimizer, restored_optimizer)

            resumed_batch = next(iter(loader))
            reloaded.train_step(resumed_batch)
            self.assertEqual(reloaded.step, 18)

    def test_periodic_checkpoint_cleanup_keeps_latest_five(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            model = create_transformer_model(vocab_size=64, d_model=32, n_heads=4, n_layers=1)
            rows = [{
                'src_tokens': torch.tensor([2, 5]),
                'tgt_input': torch.tensor([2, 6]),
                'tgt_output': torch.tensor([6, 3]),
            }]
            from torch.utils.data import Dataset
            class TinyDataset(Dataset):
                def __len__(self): return len(rows)
                def __getitem__(self, index): return rows[index]
            def tiny_collate(batch):
                return {
                    'src_tokens': torch.stack([x['src_tokens'] for x in batch]),
                    'tgt_input': torch.stack([x['tgt_input'] for x in batch]),
                    'tgt_output': torch.stack([x['tgt_output'] for x in batch]),
                }
            loader = DataLoader(TinyDataset(), batch_size=1, collate_fn=tiny_collate)
            trainer = NMTTrainer(model, loader, loader, torch.device('cpu'), vocab_size=64, checkpoint_dir=checkpoint_dir)
            for step in range(1, 8):
                trainer.step = step
                trainer.save_periodic_checkpoint()
            remaining = [name for name in os.listdir(checkpoint_dir) if name.startswith('checkpoint_step_') and name.endswith('.pt')]
            self.assertEqual(len(remaining), 5)

    def test_mid_epoch_and_completed_epoch_restore_distinguish_next_epoch(self):
        model = create_transformer_model(vocab_size=64, d_model=32, n_heads=4, n_layers=1)
        rows = [{
            'src_tokens': torch.tensor([2, 5]),
            'tgt_input': torch.tensor([2, 6]),
            'tgt_output': torch.tensor([6, 3]),
        }]
        from torch.utils.data import Dataset
        class TinyDataset(Dataset):
            def __len__(self): return len(rows)
            def __getitem__(self, index): return rows[index]
        def tiny_collate(batch):
            return {
                'src_tokens': torch.stack([x['src_tokens'] for x in batch]),
                'tgt_input': torch.stack([x['tgt_input'] for x in batch]),
                'tgt_output': torch.stack([x['tgt_output'] for x in batch]),
            }
        loader = DataLoader(TinyDataset(), batch_size=1, collate_fn=tiny_collate)

        with tempfile.TemporaryDirectory() as checkpoint_dir:
            trainer = NMTTrainer(model, loader, loader, torch.device('cpu'), vocab_size=64, checkpoint_dir=checkpoint_dir)
            trainer.epoch = 2
            trainer.step = 17

            completed_path = os.path.join(checkpoint_dir, 'completed_epoch.pt')
            trainer.save_checkpoint(completed_path, epoch_completed=True)
            reloaded = NMTTrainer(create_transformer_model(vocab_size=64, d_model=32, n_heads=4, n_layers=1), loader, loader, torch.device('cpu'), vocab_size=64, checkpoint_dir=checkpoint_dir)
            reloaded.load_checkpoint(completed_path)
            self.assertEqual(reloaded._resume_epoch, 3)

            mid_epoch_path = os.path.join(checkpoint_dir, 'interrupted_checkpoint.pt')
            trainer.save_checkpoint(mid_epoch_path, epoch_completed=False)
            reloaded.load_checkpoint(mid_epoch_path)
            self.assertEqual(reloaded._resume_epoch, 2)

    def test_cleanup_protects_best_and_recovery_checkpoints(self):
        with tempfile.TemporaryDirectory() as checkpoint_dir:
            model = create_transformer_model(vocab_size=64, d_model=32, n_heads=4, n_layers=1)
            rows = [{'src_tokens': torch.tensor([2]), 'tgt_input': torch.tensor([2]), 'tgt_output': torch.tensor([3])}]
            from torch.utils.data import Dataset
            class TinyDataset(Dataset):
                def __len__(self): return len(rows)
                def __getitem__(self, index): return rows[index]
            def tiny_collate(batch):
                return {'src_tokens': torch.stack([x['src_tokens'] for x in batch]), 'tgt_input': torch.stack([x['tgt_input'] for x in batch]), 'tgt_output': torch.stack([x['tgt_output'] for x in batch])}
            loader = DataLoader(TinyDataset(), batch_size=1, collate_fn=tiny_collate)
            trainer = NMTTrainer(model, loader, loader, torch.device('cpu'), vocab_size=64, checkpoint_dir=checkpoint_dir)

            for name in ('best_model.pt', 'interrupted_checkpoint.pt', 'error_checkpoint.pt', 'checkpoint_epoch_3.pt'):
                torch.save({}, os.path.join(checkpoint_dir, name))
            for step in range(1, 8):
                trainer.step = step
                trainer.save_periodic_checkpoint()
            trainer._cleanup_old_checkpoints()

            for name in ('best_model.pt', 'interrupted_checkpoint.pt', 'error_checkpoint.pt', 'checkpoint_epoch_3.pt'):
                self.assertTrue(os.path.exists(os.path.join(checkpoint_dir, name)))


class PhaseOneAdditionalTests(unittest.TestCase):
    def test_fully_masked_attention_row_is_zero_after_projection(self):
        torch.manual_seed(0)
        attention = MultiHeadAttention(d_model=32, n_heads=4, dropout=0.0)
        query = torch.randn(1, 3, 32)
        fully_masked = torch.zeros(1, 1, 3, 3)

        output = attention(query, query, query, fully_masked)

        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.all(output == 0))

    def test_masked_keys_and_valid_queries(self):
        torch.manual_seed(1)
        attention = MultiHeadAttention(d_model=32, n_heads=4, dropout=0.0)
        query = torch.randn(1, 2, 32)
        key = torch.randn(1, 3, 32)
        value = torch.randn(1, 3, 32)
        mask = torch.tensor([[[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]]])

        output = attention(query, key, value, mask)

        self.assertEqual(tuple(output.shape), (1, 2, 32))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.any(output != 0))

    def test_decoder_padded_positions_are_zero_and_gradients_work(self):
        model = create_transformer_model(vocab_size=128, d_model=32, n_heads=4, n_layers=2)
        model.train()
        src = torch.tensor([[2, 10, 11, 0, 0]])
        tgt = torch.tensor([[2, 20, 21, 0, 0]])
        src_mask = (src != 0).unsqueeze(1).unsqueeze(1).float()
        tgt_mask = create_target_mask(tgt)

        logits = model(src, tgt)
        loss = logits[:, :3].float().sum()
        loss.backward()

        self.assertTrue(torch.isfinite(logits).all())
        self.assertTrue(torch.equal(logits[:, 3:], torch.zeros_like(logits[:, 3:])))
        self.assertTrue(any(parameter.grad is not None for parameter in model.parameters()))

    def test_padded_keys_have_zero_attention_probability(self):
        torch.manual_seed(2)
        attention = MultiHeadAttention(d_model=32, n_heads=4, dropout=0.0)
        q = torch.randn(1, 4, 3, 8)
        k = torch.randn(1, 4, 3, 8)
        v = torch.randn(1, 4, 3, 8)
        mask = torch.tensor([[[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]]])

        _, weights = attention.scaled_dot_product_attention(q, k, v, mask)

        self.assertTrue(torch.isfinite(weights).all())
        self.assertTrue(torch.all(weights[:, :, :, 2] == 0))

    def test_valid_queries_still_attend_to_valid_keys(self):
        torch.manual_seed(3)
        attention = MultiHeadAttention(d_model=32, n_heads=4, dropout=0.0)
        q = torch.randn(1, 2, 32)
        k = torch.randn(1, 3, 32)
        v = torch.randn(1, 3, 32)
        mask = torch.tensor([[[[1.0, 1.0, 0.0], [1.0, 1.0, 0.0]]]])

        output = attention(q, k, v, mask)

        self.assertEqual(tuple(output.shape), (1, 2, 32))
        self.assertTrue(torch.isfinite(output).all())
        self.assertTrue(torch.any(output != 0))


if __name__ == '__main__':
    unittest.main()
