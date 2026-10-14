"""Summarize actual independent reviews without inventing labels for empty items."""

import argparse
import collections
import json
from pathlib import Path

JUDGMENTS = {"supports", "refutes", "insufficient"}


def summarize(rows):
    completed = []
    seen = set()
    for row in rows:
        if row.get("status") == "unreviewed":
            if row.get("judgment") or row.get("reviewer_id"):
                raise ValueError("Unreviewed item contains review fields")
            continue
        if row.get("status") != "reviewed":
            raise ValueError("Unknown review status")
        if (
            row.get("judgment") not in JUDGMENTS
            or not str(row.get("reviewer_id", "")).strip()
            or not str(row.get("rationale", "")).strip()
        ):
            raise ValueError(
                "Reviewed items require a valid judgment, reviewer ID and rationale"
            )
        key = (row["review_id"], row["reviewer_id"])
        if key in seen:
            raise ValueError("Duplicate reviewer/item")
        seen.add(key)
        completed.append(row)
    judgments = collections.Counter(r["judgment"] for r in completed)
    by_item = collections.defaultdict(list)
    for r in completed:
        by_item[r["review_id"]].append(r["judgment"])
    pairs = [
        (a, b) for v in by_item.values() for i, a in enumerate(v) for b in v[i + 1 :]
    ]
    return {
        "packet_rows": len(rows),
        "completed_reviews": len(completed),
        "reviewed_items": len(by_item),
        "reviewers": len({r["reviewer_id"] for r in completed}),
        "judgments": dict(judgments),
        "paired_reviewer_comparisons": len(pairs),
        "pairwise_agreement": sum(a == b for a, b in pairs) / len(pairs)
        if pairs
        else None,
        "accuracy": None,
        "accuracy_note": "No adjudicated independent reference judgments supplied. Construction labels are not semantic gold.",
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("packet", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    report = summarize(
        [json.loads(line) for line in args.packet.read_text().splitlines()]
    )
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))
