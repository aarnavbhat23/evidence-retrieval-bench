import unittest
from review_packet import summarize


class Reviews(unittest.TestCase):
    def test_empty_is_not_accuracy(self):
        result = summarize(
            [
                {
                    "review_id": "x",
                    "status": "unreviewed",
                    "judgment": "",
                    "reviewer_id": "",
                }
            ]
        )
        self.assertEqual(result["completed_reviews"], 0)
        self.assertIsNone(result["accuracy"])
        self.assertIsNone(result["pairwise_agreement"])

    def test_missing_reviewer_rejected(self):
        with self.assertRaises(ValueError):
            summarize(
                [
                    {
                        "review_id": "x",
                        "status": "reviewed",
                        "judgment": "supports",
                        "rationale": "source",
                    }
                ]
            )

    def test_synthetic_schema_fixture_agreement(self):
        # Schema test only; these are not actual human reviews or released benchmark labels.
        rows = [
            {
                "review_id": "x",
                "reviewer_id": r,
                "status": "reviewed",
                "judgment": j,
                "rationale": "fixture",
            }
            for r, j in [("a", "supports"), ("b", "insufficient")]
        ]
        self.assertEqual(summarize(rows)["pairwise_agreement"], 0.0)
        with self.assertRaises(ValueError):
            summarize(rows + rows[:1])
