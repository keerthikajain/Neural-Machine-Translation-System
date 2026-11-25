import unittest
from unittest.mock import patch

from saarthi_mini.pilot4_curate import diagnostic_issues
from saarthi_mini.pilot4_score import evaluate, paired_stats


class MiniPilot4ReadinessTests(unittest.TestCase):
    def test_scorer_qc(self):
        summary = paired_stats([[1, 1, -1, 0]])
        self.assertEqual((summary["wins"], summary["losses"], summary["ties"]), (2, 1, 1))
        self.assertEqual(summary["mcnemar_exact_p"], 1)
        self.assertEqual(summary["percentage_point_difference"], 25)

    def test_unresolved_judgment_fails(self):
        draft = [dict(id=f"example-{i}", category="action" if i < 30 else "quantity", tags=["negation"])
                 for i in range(60)]
        grades = [dict(id=r["id"], control={"overall": False}, treatment={"overall": None}) for r in draft]
        with self.assertRaisesRegex(ValueError, "Incomplete adjudication"):
            evaluate(draft, grades, 10.0, 10.0)

    def test_negation_check_uses_frozen_quota(self):
        draft = [dict(id=f"example-{i}", category="action" if i < 30 else "quantity",
                      tags=["negation"] if i < 9 else []) for i in range(60)]
        checked = {r["id"]: {"validation_family_rows": 0} for r in draft}
        with patch("saarthi_mini.pilot4_curate.quality", return_value=None), \
                patch("saarthi_mini.pilot4_curate.families", return_value=[]), \
                patch("saarthi_mini.pilot4_curate.overlap", return_value=checked):
            issues, _ = diagnostic_issues([], [], [], draft, None)
        self.assertEqual(issues, {"negation_quota": True})
