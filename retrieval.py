"""Lexical, frozen dense and trainable dense retrieval with explicit rankings."""

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def rank(scores, topk):
    return np.argsort(-np.asarray(scores), kind="stable")[:topk]


def assert_score_compatible_ranking(scores, ranking, tolerance=1e-6):
    """Verify descending scores and top-k inclusion, permitting numerical ties only."""
    scores = np.asarray(scores)
    indexes = np.asarray(ranking, dtype=int)
    if (
        len(set(indexes)) != len(indexes)
        or (indexes < 0).any()
        or (indexes >= len(scores)).any()
    ):
        raise AssertionError("Invalid or repeated ranked index")
    if not len(indexes) or not np.isfinite(scores).all():
        raise AssertionError("Empty or nonfinite ranking")
    if (np.diff(scores[indexes]) > tolerance).any():
        raise AssertionError("Ordering differs beyond numerical tolerance")
    excluded = np.ones(len(scores), dtype=bool)
    excluded[indexes] = False
    if excluded.any() and scores[excluded].max() - scores[indexes].min() > tolerance:
        raise AssertionError("Top-k inclusion differs beyond numerical tolerance")


def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for position, index in enumerate(ranking):
            scores[int(index)] = scores.get(int(index), 0.0) + 1 / (k + position + 1)
    return sorted(scores, key=lambda i: (-scores[i], i))


class Projection(nn.Module):
    def __init__(self, input_dim=384, output_dim=128):
        super().__init__()
        self.linear = nn.Linear(input_dim, output_dim, bias=False)

    def forward(self, values):
        return F.normalize(self.linear(values), dim=-1)


def contrastive_loss(queries, documents, document_ids, temperature=0.07):
    """Every in-batch copy of the correct document is a positive, never a negative."""
    logits = queries @ documents.T / temperature
    positives = document_ids[:, None] == document_ids[None, :]
    return (
        torch.logsumexp(logits, dim=1)
        - torch.logsumexp(logits.masked_fill(~positives, -torch.inf), dim=1)
    ).mean()


def all_positive_loss(queries, documents, positives, temperature=0.07):
    """Corpus-level loss, marginalizing every annotated positive document."""
    if positives.shape != (len(queries), len(documents)) or not positives.any(1).all():
        raise ValueError("Every query needs at least one indexed positive")
    logits = queries @ documents.T / temperature
    return (
        torch.logsumexp(logits, 1)
        - torch.logsumexp(logits.masked_fill(~positives, -torch.inf), 1)
    ).mean()


def gold_label(claim):
    labels = {e["label"] for values in claim["evidence"].values() for e in values}
    if len(labels) > 1:
        raise ValueError(
            "Mixed support/refute annotations require document-level evaluation"
        )
    return {"SUPPORT": "supported", "CONTRADICT": "contradicted"}.get(
        next(iter(labels), None), "insufficient"
    )
