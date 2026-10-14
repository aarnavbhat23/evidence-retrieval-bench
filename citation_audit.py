"""A system-independent provenance instrument. It never infers scientific support."""

import argparse
import hashlib
import json
from pathlib import Path


def content_hash(document):
    data = json.dumps(
        {"title": document["title"], "abstract": document["abstract"]},
        sort_keys=True,
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return hashlib.sha256(data.encode()).hexdigest()


def audit(record, catalog):
    if not isinstance(record.get("claim_id"), (str, int)) or isinstance(
        record.get("claim_id"), bool
    ):
        raise ValueError("claim_id must be a string/integer identifier")
    if not isinstance(record.get("text"), str) or not record["text"].strip():
        raise ValueError("Empty claim text")
    if not isinstance(record.get("abstained", False), bool):
        raise ValueError("abstained must be boolean")
    citations = record.get("citations", [])
    if not isinstance(citations, list):
        raise ValueError("citations must be a list")
    if record.get("abstained", False):
        return {
            "claim_id": record["claim_id"],
            "status": "abstained",
            "flags": [],
            "semantic_support": "not_assessed",
        }
    flags = []
    if not citations:
        flags.append("missing_citation")
    for citation in citations:
        if not isinstance(citation, dict):
            raise ValueError("Citation must be an object")
        identifier = str(citation.get("doc_id"))
        if identifier not in catalog:
            flags.append("unknown_document")
            continue
        document = catalog[identifier]
        indexes = citation.get("sentence_indexes", [])
        if (
            not isinstance(indexes, list)
            or not indexes
            or any(
                type(i) is not int or i < 0 or i >= len(document["abstract"])
                for i in indexes
            )
        ):
            flags.append("invalid_sentence_indexes")
            continue
        selected = " ".join(document["abstract"][i] for i in indexes)
        quote = citation.get("quote")
        if not isinstance(quote, str) or not quote or quote not in selected:
            flags.append("quote_not_in_selected_passage")
        supplied = citation.get("source_sha256")
        if supplied is None:
            flags.append("missing_source_hash")
        elif supplied != content_hash(document):
            flags.append("source_hash_mismatch")
    return {
        "claim_id": record["claim_id"],
        "status": "structural_flags" if flags else "structurally_consistent",
        "flags": sorted(set(flags)),
        "semantic_support": "not_assessed",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("reports", type=Path)
    parser.add_argument("--catalog", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    catalog = {
        str(r["doc_id"]): r
        for r in map(json.loads, args.catalog.read_text().splitlines())
    }
    seen = set()
    results = []
    for line in args.reports.read_text().splitlines():
        record = json.loads(line)
        if str(record.get("claim_id")) in seen:
            raise ValueError("Duplicate claim ID")
        seen.add(str(record["claim_id"]))
        results.append(audit(record, catalog))
    args.output.write_text("".join(json.dumps(r) + "\n" for r in results))
