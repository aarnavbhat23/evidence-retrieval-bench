"""All-positive retrieval with separate document-group validation/calibration."""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
from corpus import load_archive
from document_partitions import partition, references
from experiment import LABELS, NLI, NLI_REVISION, retrieve, selective_metrics, verify
from retrieval import Projection, all_positive_loss, gold_label
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score, f1_score
from transformers import AutoModelForSequenceClassification, AutoTokenizer


def fit(documents, queries, claims, lookup, labels):
    indexes = {
        s: np.array(
            [i for i, c in enumerate(claims) if labels[i] == s and c["evidence"]]
        )
        for s in ["train", "validation"]
    }
    positives = torch.zeros((len(claims), len(documents)), dtype=torch.bool)
    for i, claim in enumerate(claims):
        for identifier in claim["evidence"]:
            positives[i, lookup[int(identifier)]] = True
    dq, cq = torch.tensor(documents), torch.tensor(queries)
    torch.manual_seed(916)
    rng = np.random.default_rng(916)
    model = Projection()
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.001, weight_decay=0.01)
    best, stale, state, history = float("inf"), 0, None, []
    for epoch in range(100):
        model.train()
        losses = []
        order = rng.permutation(indexes["train"])
        for offset in range(0, len(order), 64):
            batch = order[offset : offset + 64]
            optimizer.zero_grad()
            loss = all_positive_loss(model(cq[batch]), model(dq), positives[batch])
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            idx = indexes["validation"]
            value = float(all_positive_loss(model(cq[idx]), model(dq), positives[idx]))
        history.append(
            {
                "epoch": epoch + 1,
                "train_all_positive_loss": float(np.mean(losses)),
                "validation_all_positive_loss": value,
            }
        )
        if value < best:
            best, stale, state = value, 0, copy.deepcopy(model.state_dict())
        else:
            stale += 1
        if stale == 10:
            break
    model.load_state_dict(state)
    model.eval()
    return model, history, indexes


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--device", default="mps" if torch.backends.mps.is_available() else "cpu"
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    out = Path("results/document-disjoint-v2")
    out.mkdir(parents=True, exist_ok=True)
    documents, train, dev = load_archive("data/scifact.tar.gz")
    emb = np.load("data/embeddings.npz")
    labels, groups = partition(train)
    lookup = {d["doc_id"]: i for i, d in enumerate(documents)}
    model, history, indexes = fit(emb["documents"], emb["train"], train, lookup, labels)
    torch.save(model.state_dict(), out / "projection.pt")
    (out / "history.json").write_text(json.dumps(history, indent=2) + "\n")
    split_rows = [
        {
            "claim_id": c["id"],
            "split": str(s),
            "label": gold_label(c),
            "referenced_doc_ids": references(c),
            "positive_doc_ids": sorted(map(int, c["evidence"])),
        }
        for c, s in zip(train, labels)
    ]
    (out / "split-ledger.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in split_rows)
    )
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2), min_df=2, max_features=80000, stop_words="english"
    )
    matrix = vectorizer.fit_transform(
        [d["title"] + " " + " ".join(d["abstract"]) for d in documents]
    )
    rankings = retrieve(
        dev, emb["dev"], documents, emb["documents"], vectorizer, matrix, model
    )
    training_docs = {
        d for c, s in zip(train, labels) if s == "train" for d in references(c)
    }
    train_text = {c["claim"].strip().casefold() for c in train}
    ledger = []
    retrieval = {}
    for method, lists in rankings.items():
        rows = [
            {
                "claim_id": c["id"],
                "method": method,
                "ranked_doc_ids": [documents[j]["doc_id"] for j in ranking],
                "gold_doc_ids": sorted(map(int, c["evidence"])),
                "gold_document_unreferenced_in_training_claims": not bool(
                    set(map(int, c["evidence"])) & training_docs
                ),
                "exact_train_text_duplicate": c["claim"].strip().casefold()
                in train_text,
            }
            for c, ranking in zip(dev, lists)
        ]
        ledger.extend(rows)
        retrieval[method] = {}
        for name, subset in [
            ("official_annotated_dev", [r for r in rows if r["gold_doc_ids"]]),
            (
                "unseen_gold_document_and_no_exact_text_duplicate",
                [
                    r
                    for r in rows
                    if r["gold_doc_ids"]
                    and r["gold_document_unreferenced_in_training_claims"]
                    and not r["exact_train_text_duplicate"]
                ],
            ),
        ]:
            retrieval[method][name] = {
                "claims": len(subset),
                **{
                    f"recall_at_{k}": float(
                        np.mean(
                            [
                                bool(
                                    set(r["ranked_doc_ids"][:k])
                                    & set(r["gold_doc_ids"])
                                )
                                for r in subset
                            ]
                        )
                    )
                    for k in [1, 5, 10, 50]
                },
            }
    (out / "retrieval-ledger.jsonl").write_text(
        "".join(json.dumps(r) + "\n" for r in ledger)
    )
    tokenizer = AutoTokenizer.from_pretrained(NLI, revision=NLI_REVISION)
    verifier = (
        AutoModelForSequenceClassification.from_pretrained(NLI, revision=NLI_REVISION)
        .to(args.device)
        .eval()
    )
    results, truncated = verify(
        dev,
        rankings["hybrid_rrf"],
        documents,
        vectorizer,
        tokenizer,
        verifier,
        args.device,
    )
    cal_idx = np.flatnonzero(labels == "calibration")
    cal_claims = [train[i] for i in cal_idx]
    cr = retrieve(
        cal_claims,
        emb["train"][cal_idx],
        documents,
        emb["documents"],
        vectorizer,
        matrix,
        model,
    )
    cal, cal_truncated = verify(
        cal_claims,
        cr["hybrid_rrf"],
        documents,
        vectorizer,
        tokenizer,
        verifier,
        args.device,
    )
    thresholds = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99, 1.0]
    curve = [selective_metrics(cal, t) for t in thresholds]
    eligible = [
        r
        for r in curve
        if r["accepted"] >= 10 and r["label_and_citation_precision"] >= 0.9
    ]
    threshold = min(r["threshold"] for r in eligible) if eligible else 1.0
    for name, rows in [("verification", results), ("calibration-verification", cal)]:
        (out / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    report = {
        "protocol": "docs/EXTENSION_PROTOCOL.md",
        "training_claims_by_split": {
            s: {
                "claims": int((labels == s).sum()),
                "annotated": sum(
                    bool(c["evidence"]) for c, label in zip(train, labels) if label == s
                ),
                "insufficient": sum(
                    not c["evidence"] for c, label in zip(train, labels) if label == s
                ),
            }
            for s in ["train", "validation", "calibration"]
        },
        "components": len(groups),
        "all_reference_group_overlap": 0,
        "projection_positive_counts": {s: len(i) for s, i in indexes.items()},
        "selected_epoch": min(history, key=lambda h: h["validation_all_positive_loss"])[
            "epoch"
        ],
        "retrieval": retrieval,
        "verification": {
            "nli_model": NLI,
            "revision": NLI_REVISION,
            "dev_claims": len(results),
            "truncated_dev_pairs": truncated,
            "truncated_calibration_pairs": cal_truncated,
            "accuracy": float(
                accuracy_score(
                    [r["gold"] for r in results], [r["prediction"] for r in results]
                )
            ),
            "macro_f1": float(
                f1_score(
                    [r["gold"] for r in results],
                    [r["prediction"] for r in results],
                    labels=LABELS,
                    average="macro",
                )
            ),
            "calibration_risk_coverage": curve,
            "development_risk_coverage": [
                selective_metrics(results, t) for t in thresholds
            ],
            "chosen_policy": selective_metrics(results, threshold),
        },
        "limitations": [
            "Empirical policy precision is not a safety guarantee.",
            "NEI claims are retained in calibration but have no positive retrieval training target.",
            "Official development results are not a new independent corpus; unseen-document stratum is a diagnostic.",
            "Frozen upstream pretraining contamination is unknown.",
            "NLI predictions are not human ground truth.",
        ],
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
