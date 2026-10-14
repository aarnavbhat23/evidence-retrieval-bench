"""Paired corruption stress tests and a genuine unreviewed human-review packet."""

import copy
import hashlib
import json
from pathlib import Path

import numpy as np
from citation_audit import audit, content_hash
from corpus import load_archive


def build_cases(documents, claims):
    catalog = {str(d["doc_id"]): d for d in documents}
    keys = sorted(catalog, key=int)
    records = []
    for claim in claims:
        if not claim["evidence"]:
            continue
        identifier = sorted(claim["evidence"], key=int)[0]
        doc = catalog[identifier]
        indexes = claim["evidence"][identifier][0]["sentences"]
        quote = " ".join(doc["abstract"][i] for i in indexes)
        original = {
            "claim_id": f"{claim['id']}/original",
            "text": claim["claim"],
            "abstained": False,
            "citations": [
                {
                    "doc_id": int(identifier),
                    "sentence_indexes": indexes,
                    "quote": quote,
                    "source_sha256": content_hash(doc),
                }
            ],
        }
        variants = {"original": original}
        altered = copy.deepcopy(original)
        altered["citations"][0]["quote"] = "[ALTERED QUOTE] " + quote
        variants["altered_quote"] = altered
        missing = copy.deepcopy(original)
        missing["citations"] = []
        variants["missing_citation"] = missing
        invalid = copy.deepcopy(original)
        invalid["citations"][0]["sentence_indexes"] = [len(doc["abstract"])]
        variants["invalid_index"] = invalid
        mismatch = copy.deepcopy(original)
        mismatch["citations"][0]["source_sha256"] = "0" * 64
        variants["hash_mismatch"] = mismatch
        other = next(
            k for k in keys if k not in claim["evidence"] and catalog[k]["abstract"]
        )
        swapped = copy.deepcopy(original)
        wrong = catalog[other]
        swapped["citations"] = [
            {
                "doc_id": int(other),
                "sentence_indexes": [0],
                "quote": wrong["abstract"][0],
                "source_sha256": content_hash(wrong),
            }
        ]
        variants["consistent_wrong_source"] = swapped
        for name, record in variants.items():
            record["claim_id"] = f"{claim['id']}/{name}"
            result = audit(record, catalog)
            records.append(
                {
                    "source_claim_id": claim["id"],
                    "variant": name,
                    "record": record,
                    "audit": result,
                    "citation_matches_expert_annotation": name
                    not in ["missing_citation", "consistent_wrong_source"],
                    "expected_structural_flag": name
                    not in ["original", "consistent_wrong_source"],
                }
            )
    return records, catalog


def main():
    documents, _, dev = load_archive("data/scifact.tar.gz")
    cases, catalog = build_cases(documents, dev)
    out = Path("results/citation-challenge")
    out.mkdir(parents=True, exist_ok=True)
    (out / "cases.jsonl").write_text("".join(json.dumps(c) + "\n" for c in cases))
    summaries = {}
    for name in sorted({c["variant"] for c in cases}):
        rows = [c for c in cases if c["variant"] == name]
        summaries[name] = {
            "cases": len(rows),
            "structural_flag_fraction": float(
                np.mean([bool(c["audit"]["flags"]) for c in rows])
            ),
            "annotated_citation_match_fraction": float(
                np.mean([c["citation_matches_expert_annotation"] for c in rows])
            ),
        }
    report = {
        "independent_source_claims": len({c["source_claim_id"] for c in cases}),
        "paired_variants": len(cases),
        "variants": summaries,
        "human_review_status": "unreviewed",
        "note": "Controlled edits, not a deployed-system error distribution. Semantic entailment is never established by structural consistency.",
    }
    (out / "metrics.json").write_text(json.dumps(report, indent=2) + "\n")
    # Blinded packet omits model output, construction labels, and expert gold labels.
    rng = np.random.default_rng(919)
    chosen = rng.choice(len(cases), size=60, replace=False)
    packet = []
    key = []
    for i in chosen:
        case = cases[int(i)]
        record = copy.deepcopy(case["record"])
        token = hashlib.sha256(f"919/{record['claim_id']}".encode()).hexdigest()[:12]
        packet.append(
            {
                "review_id": token,
                "claim": record["text"],
                "citations": record["citations"],
                "sources": [catalog[str(c["doc_id"])] for c in record["citations"]],
                "reviewer_id": "",
                "judgment": "",
                "rationale": "",
                "status": "unreviewed",
            }
        )
        key.append(
            {
                "review_id": token,
                "source_claim_id": case["source_claim_id"],
                "variant": case["variant"],
                "citation_matches_expert_annotation": case[
                    "citation_matches_expert_annotation"
                ],
            }
        )
    (out / "blind-review-packet.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in packet)
    )
    (out / "review-construction-key.jsonl").write_text(
        "".join(json.dumps(c) + "\n" for c in key)
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
