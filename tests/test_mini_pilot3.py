import unittest

from saarthi_mini.pilot3 import BASE_NLL, divergence_streak
from saarthi_mini.pilot3_compare import blind_packet, select_step
from saarthi_mini.train import learning_rate


class MiniPilot3Tests(unittest.TestCase):
    def test_continuation_rate_changes_only_after_fork(self):
        base = {"warmup_updates": 60, "max_updates": 600, "peak_lr": 0.001}
        control = dict(base, max_updates=1650, schedule_updates=600, continuation_lr=0.0001)
        treatment = dict(control, continuation_lr=0.0003)
        for step in (1, 60, 599, 600):
            self.assertEqual(learning_rate(step, base), learning_rate(step, control))
            self.assertEqual(learning_rate(step, base), learning_rate(step, treatment))
        for step in (601, 750, 1650):
            self.assertEqual(learning_rate(step, control), 0.0001)
            self.assertEqual(learning_rate(step, treatment), 0.0003)

    def test_divergence_requires_two_consecutive_strict_exceedances(self):
        self.assertEqual(divergence_streak(1, BASE_NLL + 0.3), 0)
        self.assertEqual(divergence_streak(0, BASE_NLL + 0.31), 1)
        self.assertEqual(divergence_streak(1, BASE_NLL + 0.31), 2)

    def test_selection_reuses_full_precision_patience_threshold(self):
        checks = {750: {"chrf++": 11.49}, 900: {"chrf++": 11.5},
                  1050: {"chrf++": 11.999}, 1200: {"chrf++": 12.0}}
        self.assertEqual(select_step(checks, 11.0, 0.5), 1200)
        self.assertIsNone(select_step({750: checks[750]}, 11.0, 0.5))

    def test_blind_packet_omits_arm_labels_and_keeps_separate_key(self):
        sample = [{"id": "pair", "stratum": "other", "en": "Example", "refs": ["देखो"]}]
        predictions = {"control": {"pair": {"prediction": "एक"}},
                       "treatment": {"pair": {"prediction": "दो"}}}
        packet, key = blind_packet(sample, predictions)
        self.assertEqual(set(packet[0]), {"item", "english", "references", "A", "B",
                                          "A_adequate", "B_adequate", "notes"})
        for label in ("A", "B"):
            self.assertEqual(packet[0][label], predictions[key["1"][label]]["pair"]["prediction"])
