"""Reproduce document-group boundaries, released rankings and citation challenge."""

import json
from pathlib import Path

import numpy as np
import torch
from citation_challenge import build_cases
from corpus import load_archive
from document_partitions import partition, references
from experiment import LABELS, retrieve, selective_metrics
from retrieval import Projection
from review_packet import summarize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score


def read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines()]


def main():
    torch.set_num_threads(4)
    out = Path("results/document-disjoint-v2")
    report = json.loads((out / "metrics.json").read_text())
    split = read(out / "split-ledger.jsonl")
    documents, train, dev = load_archive("data/scifact.tar.gz")
    labels, _ = partition(train)
    assert [r["split"] for r in split] == labels.tolist()
    for claim, row in zip(train, split):
        assert row["referenced_doc_ids"] == references(claim)
    by_split = {
        s: {d for r in split if r["split"] == s for d in r["referenced_doc_ids"]}
        for s in set(labels)
    }
    for a in by_split:
        for b in by_split:
            if a != b:
                assert not by_split[a] & by_split[b]
    ledger = read(out / "retrieval-ledger.jsonl")
    for method, strata in report["retrieval"].items():
        rows = [r for r in ledger if r["method"] == method and r["gold_doc_ids"]]
        for name, expected in strata.items():
            subset = (
                rows
                if name == "official_annotated_dev"
                else [
                    r
                    for r in rows
                    if r["gold_document_unreferenced_in_training_claims"]
                    and not r["exact_train_text_duplicate"]
                ]
            )
            assert len(subset) == expected["claims"]
            for k in [1, 5, 10, 50]:
                np.testing.assert_allclose(
                    np.mean(
                        [
                            bool(set(r["ranked_doc_ids"][:k]) & set(r["gold_doc_ids"]))
                            for r in subset
                        ]
                    ),
                    expected[f"recall_at_{k}"],
                )
    emb = np.load("data/embeddings.npz")
    projection = Projection().eval()
    projection.load_state_dict(
        torch.load(out / "projection.pt", weights_only=True, map_location="cpu")
    )
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=80000, stop_words="english"
    )
    matrix = vectorizer.fit_transform(
        [d["title"] + " " + " ".join(d["abstract"]) for d in documents]
    )
    regenerated = retrieve(
        dev, emb["dev"], documents, emb["documents"], vectorizer, matrix, projection
    )
    for method, rankings in regenerated.items():
        rows = [r for r in ledger if r["method"] == method]
        for row, ranking in zip(rows, rankings):
            assert row["ranked_doc_ids"] == [documents[j]["doc_id"] for j in ranking]
    expected = report["verification"]
    results = read(out / "verification.jsonl")
    cal = read(out / "calibration-verification.jsonl")
    assert (
        len(results) == 300
        and len(cal) == report["training_claims_by_split"]["calibration"]["claims"]
    )
    np.testing.assert_allclose(
        accuracy_score(
            [r["gold"] for r in results], [r["prediction"] for r in results]
        ),
        expected["accuracy"],
    )
    np.testing.assert_allclose(
        f1_score(
            [r["gold"] for r in results],
            [r["prediction"] for r in results],
            labels=LABELS,
            average="macro",
        ),
        expected["macro_f1"],
    )
    for rows, name in [
        (cal, "calibration_risk_coverage"),
        (results, "development_risk_coverage"),
    ]:
        for metric in expected[name]:
            assert selective_metrics(rows, metric["threshold"]) == metric
    eligible = [
        r
        for r in expected["calibration_risk_coverage"]
        if r["accepted"] >= 10 and r["label_and_citation_precision"] >= 0.9
    ]
    threshold = min(r["threshold"] for r in eligible) if eligible else 1.0
    assert selective_metrics(results, threshold) == expected["chosen_policy"]
    cases, _ = build_cases(documents, dev)
    assert cases == read("results/citation-challenge/cases.jsonl")
    packet = read("results/citation-challenge/blind-review-packet.jsonl")
    assert summarize(packet) == json.loads(
        Path("results/citation-challenge/review-status.json").read_text()
    )
    print(
        "Replayed all-positive split groups, four fitted retrieval rankings, 300 verdict metrics, 1128 paired citation cases and unreviewed packet status."
    )


if __name__ == "__main__":
    main()
