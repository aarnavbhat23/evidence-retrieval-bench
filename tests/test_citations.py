import unittest
import copy
import torch
from citation_audit import audit, content_hash
from document_partitions import partition
from retrieval import all_positive_loss


class CitationInstrumentTests(unittest.TestCase):
    def setUp(self):
        self.doc = {
            "doc_id": 1,
            "title": "Test",
            "abstract": ["A measured observation."],
        }
        self.catalog = {"1": self.doc}
        self.record = {
            "claim_id": "x",
            "text": "Claim",
            "citations": [
                {
                    "doc_id": 1,
                    "sentence_indexes": [0],
                    "quote": "A measured observation.",
                    "source_sha256": content_hash(self.doc),
                }
            ],
        }

    def test_valid_provenance_is_not_semantic_verification(self):
        result = audit(self.record, self.catalog)
        self.assertEqual(result["status"], "structurally_consistent")
        self.assertEqual(result["semantic_support"], "not_assessed")

    def test_altered_quote(self):
        self.record["citations"][0]["quote"] = "Invented observation."
        self.assertIn(
            "quote_not_in_selected_passage", audit(self.record, self.catalog)["flags"]
        )

    def test_boolean_index_rejected(self):
        self.record["citations"][0]["sentence_indexes"] = [False]
        self.assertIn(
            "invalid_sentence_indexes", audit(self.record, self.catalog)["flags"]
        )

    def test_unavailable_is_not_refutation(self):
        self.record["citations"][0]["doc_id"] = 9
        result = audit(self.record, self.catalog)
        self.assertIn("unknown_document", result["flags"])
        self.assertEqual(result["semantic_support"], "not_assessed")

    def test_abstentions_separate(self):
        self.record["abstained"] = True
        self.record["citations"] = []
        self.assertEqual(audit(self.record, self.catalog)["status"], "abstained")

    def test_all_positive_loss(self):
        query = torch.tensor([[1.0, 0.0]])
        docs = torch.tensor([[1.0, 0.0], [1.0, 0.0]])
        positives = torch.tensor([[True, True]])
        self.assertAlmostEqual(all_positive_loss(query, docs, positives).item(), 0.0)

    def test_secondary_positive_does_not_leak(self):
        claims = [
            {"cited_doc_ids": [1, 2], "evidence": {"1": []}},
            {"cited_doc_ids": [2], "evidence": {"2": []}},
            {"cited_doc_ids": [3], "evidence": {}},
            {"cited_doc_ids": [4], "evidence": {}},
        ]
        labels, _ = partition(claims)
        self.assertEqual(labels[0], labels[1])


if __name__ == "__main__":
    unittest.main()
