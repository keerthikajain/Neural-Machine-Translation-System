import json
from pathlib import Path
import unittest

from saarthi_mini.train import learning_rate


class MiniContinuationTests(unittest.TestCase):
    def test_pilot2_keeps_every_pilot1_learning_rate(self):
        directory = Path(__file__).resolve().parents[1] / "saarthi_mini"
        parent, continuation = (
            json.loads((directory / filename).read_text(encoding="utf-8"))
            for filename in ("pilot.json", "pilot2.json")
        )
        prefix = parent["max_updates"]
        self.assertEqual(continuation["schedule_updates"], prefix)
        self.assertEqual(
            [learning_rate(update, parent) for update in range(1, prefix + 1)],
            [learning_rate(update, continuation) for update in range(1, prefix + 1)],
        )
        floor = learning_rate(prefix, parent)
        self.assertEqual(
            {learning_rate(update, continuation)
             for update in range(prefix + 1, continuation["max_updates"] + 1)},
            {floor},
        )
