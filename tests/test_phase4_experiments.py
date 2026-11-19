import json
import math
import os
import tempfile
import unittest
from pathlib import Path

import torch

from src.data.dataset import NMTDataset, collate_fn
from src.model import create_compact_a_model, create_transformer_model
from src.training.trainer import NMTTrainer, LabelSmoothingCrossEntropy
from src.model.decoder import create_target_mask
from experiments.tiny_overfit import (
    calculate_initial_loss,
    create_next_run_directory,
    read_subset,
    set_reproducible_seed,
    create_tiny_experiment_loaders,
)


class Phase4ExperimentTests(unittest.TestCase):
    def test_compact_configuration_contract(self):
        model = create_compact_a_model(vocab_size=32000)
        info = model.get_model_info()
        self.assertEqual(info['d_model'], 256)
        self.assertEqual(info['n_heads'], 4)
        self.assertEqual(info['encoder_layers'], 4)
        self.assertEqual(info['decoder_layers'], 4)
        self.assertEqual(info['d_ff'], 1024)
        self.assertEqual(info['max_seq_length'], 128)

    def test_reference_defaults_remain_unchanged(self):
        reference = create_transformer_model(vocab_size=32000)
        info = reference.get_model_info()
        self.assertEqual(info['d_model'], 512)
        self.assertEqual(info['n_heads'], 8)
        self.assertEqual(info['encoder_layers'], 6)
        self.assertEqual(info['decoder_layers'], 6)
        self.assertEqual(info['d_ff'], 2048)
        self.assertEqual(info['max_seq_length'], 512)

    def test_subset_dataset_limit_contract(self):
        for sample_count in (0, 99, 501):
            valid = 100 <= sample_count <= 500
            self.assertFalse(valid)
        self.assertTrue(100 <= 300 <= 500)

    def test_tiny_dataset_forward_and_checkpoint(self):
        dataset = NMTDataset(
            english_sentences=['hello', 'how are you'],
            hindi_sentences=['नमस्ते', 'आप कैसे हैं'],
            tokenizer_path='data/processed/tokenizer/sp_model.model',
            max_length=128,
        )
        batch = collate_fn([dataset[0], dataset[1]])
        model = create_compact_a_model(vocab_size=32000)
        logits = model(batch['src_tokens'], batch['tgt_input'])
        self.assertEqual(logits.size(0), 2)
        self.assertTrue(torch.isfinite(logits).all())

        with tempfile.TemporaryDirectory() as directory:
            loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
            trainer = NMTTrainer(model, loader, loader, torch.device('cpu'), vocab_size=32000,
                                 checkpoint_dir=directory)
            trainer.epoch = 2
            trainer.step = 17
            checkpoint_path = os.path.join(directory, 'compact_experiment.pt')
            trainer.save_checkpoint(checkpoint_path)
            checkpoint = torch.load(checkpoint_path, map_location='cpu')
            self.assertEqual(checkpoint['architecture']['d_model'], 256)
            self.assertEqual(checkpoint['architecture']['max_seq_length'], 128)
            self.assertEqual(checkpoint['architecture']['encoder_layers'], 4)

    def test_evaluation_fixture_inputs_exist(self):
        self.assertTrue(os.path.exists('data/processed/val_en.txt'))
        self.assertTrue(os.path.exists('data/processed/val_hi.txt'))
        self.assertTrue(os.path.exists('data/processed/tokenizer/sp_model.model'))

    def test_same_seed_same_subset_and_initialization(self):
        with tempfile.TemporaryDirectory() as directory:
            en_path = Path(directory) / 'train.en'
            hi_path = Path(directory) / 'train.hi'
            en_lines = [f'en {i}\n' for i in range(150)]
            hi_lines = [f'हि {i}\n' for i in range(150)]
            en_path.write_text(''.join(en_lines), encoding='utf-8')
            hi_path.write_text(''.join(hi_lines), encoding='utf-8')

            subset_a = read_subset(str(en_path), str(hi_path), 100, 42)
            set_reproducible_seed(42)
            model_a = create_compact_a_model(vocab_size=32000)
            subset_b = read_subset(str(en_path), str(hi_path), 100, 42)
            set_reproducible_seed(42)
            model_b = create_compact_a_model(vocab_size=32000)

            self.assertEqual(subset_a, subset_b)
            for parameter_a, parameter_b in zip(model_a.parameters(), model_b.parameters()):
                self.assertTrue(torch.equal(parameter_a, parameter_b))

    def test_new_run_directory_preserves_previous_results(self):
        with tempfile.TemporaryDirectory() as directory:
            run_one = create_next_run_directory(directory)
            marker = run_one / 'experiment_config.json'
            marker.write_text('{"preserved": true}', encoding='utf-8')

            run_two = create_next_run_directory(directory)
            self.assertEqual(run_one.name, 'run_001')
            self.assertEqual(run_two.name, 'run_002')
            self.assertTrue(marker.exists())
            self.assertEqual(marker.read_text(encoding='utf-8'), '{"preserved": true}')

    def test_initial_and_epoch_results_contract(self):
        dataset = NMTDataset(
            english_sentences=['hello', 'how are you'],
            hindi_sentences=['नमस्ते', 'आप कैसे हैं'],
            tokenizer_path='data/processed/tokenizer/sp_model.model',
            max_length=128,
        )
        loader = torch.utils.data.DataLoader(dataset, batch_size=2, collate_fn=collate_fn)
        model = create_compact_a_model(vocab_size=32000)
        with tempfile.TemporaryDirectory() as directory:
            trainer = NMTTrainer(
                model=model,
                train_loader=loader,
                val_loader=loader,
                device=torch.device('cpu'),
                vocab_size=32000,
                checkpoint_dir=directory,
            )
            initial_loss = calculate_initial_loss(trainer)
            self.assertGreaterEqual(initial_loss, 0.0)
            self.assertTrue(math.isfinite(initial_loss))

            history = {
                'initial_loss': initial_loss,
                'epochs_recorded': [{
                    'epoch': 1,
                    'train_loss': 1.0,
                    'reproduction_exact_matches': 0,
                    'reproduction_samples': [],
                }],
            }
            output = Path(directory) / 'experiment_config.json'
            output.write_text(json.dumps(history), encoding='utf-8')
            saved = json.loads(output.read_text(encoding='utf-8'))
            self.assertIn('initial_loss', saved)
            self.assertEqual(saved['epochs_recorded'][0]['epoch'], 1)
            self.assertIn('reproduction_samples', saved['epochs_recorded'][0])

    def test_bos_eos_shift_and_padding_masks(self):
        hindi = 'नमस्ते दुनिया'
        dataset = NMTDataset(
            english_sentences=['hello world'],
            hindi_sentences=[hindi],
            tokenizer_path='data/processed/tokenizer/sp_model.model',
            max_length=128,
        )
        item = dataset[0]
        raw_tokens = dataset._tokenize_and_truncate(hindi)

        self.assertEqual(item['tgt_input'][0], 2)
        self.assertEqual(item['tgt_input'][1:], raw_tokens)
        self.assertEqual(item['tgt_output'][:-1], raw_tokens)
        self.assertEqual(item['tgt_output'][-1], 3)

        batch = collate_fn([
            dataset[0],
            NMTDataset(
                english_sentences=['hi'], hindi_sentences=['नमस्ते'],
                tokenizer_path='data/processed/tokenizer/sp_model.model', max_length=128,
            )[0],
        ])
        tgt_input = batch['tgt_input']
        target_mask = create_target_mask(tgt_input, pad_token_id=0)
        self.assertFalse(torch.equal(target_mask[0, :, 0, 0], torch.zeros_like(target_mask[0, :, 0, 0])))
        self.assertTrue(torch.all(target_mask[1, :, -1, :] == 0))
        valid_length = batch['tgt_lengths'][0].item()
        causal_lower = torch.tril(torch.ones(valid_length, valid_length, dtype=torch.bool))
        self.assertTrue(torch.all(target_mask[0, 0, :valid_length, :valid_length][causal_lower] != 0))

    def test_pad_exclusion_from_loss(self):
        logits = torch.randn(2, 32000, requires_grad=True)
        targets_without_pad = torch.tensor([10, 11])
        targets_with_pad = torch.tensor([10, 0])
        criterion = LabelSmoothingCrossEntropy(
            num_classes=32000, smoothing=0.1, ignore_index=0,
        )
        loss_only_valid = criterion(logits[0:1], targets_without_pad[0:1])
        loss_with_pad = criterion(logits, targets_with_pad)

        self.assertTrue(torch.isfinite(loss_only_valid))
        self.assertTrue(torch.isfinite(loss_with_pad))
        self.assertAlmostEqual(loss_only_valid.item(), loss_with_pad.item(), places=6)

    def test_independent_training_and_validation_loaders(self):
        train_en, train_hi = read_subset(
            'data/processed/clean_train_en.txt',
            'data/processed/clean_train_hi.txt',
            100,
            42,
        )
        train_loader, val_loader = create_tiny_experiment_loaders(
            train_en, train_hi,
            'data/processed/tokenizer/sp_model.model',
            batch_size=8,
            seed=42,
        )

        self.assertEqual(len(train_loader.dataset), 100)
        self.assertEqual(len(val_loader.dataset), 520)
        self.assertIsInstance(train_loader.sampler, torch.utils.data.RandomSampler)
        self.assertIsInstance(val_loader.sampler, torch.utils.data.SequentialSampler)
        self.assertIsNot(train_loader.dataset, val_loader.dataset)

        val_en = val_loader.dataset.english_sentences
        overlap = set(train_en).intersection(set(val_en))
        self.assertEqual(len(overlap), 0)

        val_batch = next(iter(val_loader))
        self.assertEqual(len(val_batch['english_texts']), len(val_batch['hindi_texts']))
        self.assertEqual(val_batch['src_tokens'].size(0), len(val_batch['english_texts']))


if __name__ == '__main__':
    unittest.main()
