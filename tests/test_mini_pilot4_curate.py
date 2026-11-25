import unittest

from saarthi_mini.pilot4_curate import CHECKS, approved, identity, overlap


class MiniPilot4CurationTests(unittest.TestCase):
    def test_approval_requires_two_distinct_reviews_bound_to_the_pair(self):
        pair = {"en": "The cat sat.", "hi": "बिल्ली बैठी।"}
        digest = identity(pair)
        reviews = [dict(reviewer=name, decision="approve", pair_sha256=digest,
                        checks=dict.fromkeys(CHECKS, "pass")) for name in ("Reviewer A", "Reviewer B")]
        record = dict(status="approved", pair_sha256=digest, approvals=reviews)
        self.assertTrue(approved(record))
        self.assertFalse(approved(dict(record, status="pending")))
        self.assertFalse(approved(dict(record, pair_sha256=identity(dict(pair, en="The cat ran.")))))
        self.assertFalse(approved(dict(record, approvals=[reviews[0], reviews[0]])))

    def test_validation_and_diagnostic_overlaps_are_distinct(self):
        validation = [{"id": "v", "en": "She filled three jars.", "hi": "उसने तीन मर्तबान भरे।"}]
        candidates = [{"id": "c", "en": "She filled four jars.", "hi": "उसने चार मर्तबान भरे।"}]
        diagnostic = [{"id": "d", "en": "The dog found the shoe.", "hi": "कुत्ते को जूता मिला।"}]
        train = [dict(diagnostic[0], id="t")]
        flags = overlap(train, validation, candidates, diagnostic)
        self.assertEqual(flags["c"]["validation_family_rows"], 1)
        self.assertEqual(flags["c"]["diagnostic_family_rows"], 0)
        self.assertEqual(flags["d"]["train_or_candidate_family_rows"], 1)
