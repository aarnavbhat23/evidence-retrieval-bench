import unittest
import numpy as np
import torch
from retrieval import rank,reciprocal_rank_fusion,contrastive_loss,gold_label
from experiment import selective_metrics


class RetrievalTests(unittest.TestCase):
    def test_stable_ranks(self):np.testing.assert_array_equal(rank([1,1,0],2),[0,1])

    def test_fusion(self):self.assertEqual(reciprocal_rank_fusion([[0,1],[1,0]])[:2],[0,1])

    def test_duplicate_positives_are_not_negatives(self):
        vector=torch.tensor([[1.,0.],[1.,0.]])
        self.assertAlmostEqual(contrastive_loss(vector,vector,torch.tensor([5,5])).item(),0.)

    def test_citation_required(self):
        row={'gold':'supported','prediction':'supported','confidence':.99,'citation_is_annotated_evidence':False}
        metrics=selective_metrics([row],.9)
        self.assertEqual(metrics['label_precision'],1.)
        self.assertEqual(metrics['label_and_citation_precision'],0.)

    def test_neutral_has_no_evidence(self):self.assertEqual(gold_label({'evidence':{}}),'insufficient')


if __name__=='__main__':unittest.main()
