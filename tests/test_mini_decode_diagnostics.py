from types import SimpleNamespace
import unittest

import torch

from saarthi_mini.decode_diagnostics import block_trigram, candidates, lengths_and_eos, select
from saarthi_mini.metrics import BOS, EOS, PAD, generate


class Tokenizer:
    def encode(self, text):
        return [4]

    def decode(self, ids):
        return " ".join(map(str, ids))


class ScriptedModel:
    """Greedy takes token 4, while token 5 has the likelier completed path."""
    def __init__(self):
        self.encoder = SimpleNamespace(embedding=SimpleNamespace(positional_encoding=SimpleNamespace(
            pe=torch.zeros(1, 16, 1))))

    def eval(self):
        return self

    def encode(self, source):
        return source.float(), source.ne(PAD)[:, None, None]

    def decode_step(self, prefix, memory, mask):
        logits = torch.full((*prefix.shape, 8), -20.0)
        if prefix.size(1) == 1:
            logits[:, -1, 4] = 1.0
            logits[:, -1, 5] = 0.9
            return logits
        for row, tokens in enumerate(prefix.tolist()):
            if tokens[-1] == 5 or len(tokens) >= 3:
                logits[row, -1, EOS] = 10.0
            else:
                logits[row, -1, EOS] = 0.01
                logits[row, -1, 6] = 0.0
                logits[row, -1, 7] = -0.01
        return logits


class MiniDecodeDiagnosticsTests(unittest.TestCase):
    def setUp(self):
        self.model = ScriptedModel()
        self.tokenizer = Tokenizer()

    def test_width_one_matches_original_greedy(self):
        expected = generate(self.model, self.tokenizer, ["a", "b"], "cpu", 6)
        self.assertEqual(expected, select(self.tokenizer, candidates(
            self.model, self.tokenizer, ["a", "b"], 6, width=1)))
        self.assertEqual(expected[0], {"prediction": "4", "eos": True})

    def test_beam_keeps_shorter_better_path_and_freezes_eos(self):
        result = list(candidates(self.model, self.tokenizer, ["a"], 6, width=3))
        self.assertEqual(select(self.tokenizer, result), [{"prediction": "5", "eos": True}])
        tokens, scores = result[0]
        chosen = tokens[0, scores[0].argmax()].tolist()
        self.assertEqual(chosen[:2], [5, EOS])
        self.assertTrue(all(token == PAD for token in chosen[2:]))

    def test_length_reranking_changes_ranking_without_touching_candidates(self):
        sequences = torch.tensor([[[4, EOS, PAD, PAD], [5, 6, 7, EOS]]])
        scores = torch.tensor([[-2.0, -2.3]])
        pool = [(sequences, scores)]
        self.assertEqual(select(self.tokenizer, pool, 0)[0]["prediction"], "4")
        self.assertEqual(select(self.tokenizer, pool, 0.7)[0]["prediction"], "5 6 7")
        torch.testing.assert_close(scores, torch.tensor([[-2.0, -2.3]]))

    def test_repetition_block_masks_only_previously_seen_completion(self):
        logits = torch.zeros(2, 8)
        block_trigram(logits, torch.tensor([[BOS, 4, 5, 6, 4, 5], [BOS, 4, 5, 6, 5, 4]]))
        self.assertTrue(torch.isneginf(logits[0, 6]))
        self.assertEqual(torch.isneginf(logits).sum().item(), 1)

    def test_minimum_delays_eos_and_maximum_does_not_force_it(self):
        delayed = select(self.tokenizer, candidates(
            self.model, self.tokenizer, ["a"], 5, minimum=2))
        self.assertEqual(delayed[0], {"prediction": "4 6", "eos": True})
        short = select(self.tokenizer, candidates(self.model, self.tokenizer, ["a"], 1))
        self.assertEqual(short[0], {"prediction": "4", "eos": False})
        lengths, eos = lengths_and_eos(torch.tensor([[[4, 6], [EOS, PAD]]]))
        self.assertEqual(lengths.tolist(), [[2, 1]])
        self.assertEqual(eos.tolist(), [[False, True]])
