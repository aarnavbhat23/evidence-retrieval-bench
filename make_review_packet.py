"""Render the unreviewed JSONL packet into a printable file and blank review sheet."""

import csv
import json
from html import escape
from pathlib import Path


def main():
    out = Path("results/citation-challenge")
    rows = [
        json.loads(line)
        for line in (out / "blind-review-packet.jsonl").read_text().splitlines()
    ]
    sections = []
    for i, row in enumerate(rows, 1):
        citations = (
            "".join(
                "<li><p>" + escape(json.dumps(c, ensure_ascii=False)) + "</p></li>"
                for c in row["citations"]
            )
            or "<li>No citation supplied.</li>"
        )
        sources = "".join(
            "<h3>"
            + escape(str(s["doc_id"]) + " — " + s["title"])
            + '</h3><ol start="0">'
            + "".join("<li>" + escape(t) + "</li>" for t in s["abstract"])
            + "</ol>"
            for s in row["sources"]
        )
        sections.append(
            f'<article><h2>Item {i} · {escape(row["review_id"])}</h2><p class="claim">{escape(row["claim"])}</p><h3>Citations</h3><ul>{citations}</ul>{sources}<p>Judgment: supports / refutes / insufficient<br>Rationale: __________________________</p></article>'
        )
    html = (
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>Blinded scientific source review</title><style>body{max-width:850px;margin:40px auto;padding:20px;font:17px/1.6 Georgia,serif;color:#17212d}h1,h2,h3{font-family:system-ui}h3{font-size:17px}article{border-top:2px solid #9cafbb;padding:24px 0;break-before:page}li{overflow-wrap:anywhere}.claim{font-size:21px;background:#edf3f5;padding:15px}@media print{body{margin:0;font-size:12px}article{break-before:page}}</style><h1>Blinded scientific source review</h1><p>60 unreviewed items. Judge support from the supplied sources, not general truth from memory. Review independently and retain each item ID. Read REVIEW_INSTRUCTIONS.md first. No labels or model scores are supplied here.</p>'
        + "".join(sections)
        + "</html>"
    )
    (out / "review-packet.html").write_text(html)
    with (out / "review-sheet.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "review_id",
                "reviewer_id",
                "judgment",
                "rationale",
                "quote_integrity",
                "status",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "review_id": row["review_id"],
                    "reviewer_id": "",
                    "judgment": "",
                    "rationale": "",
                    "quote_integrity": "",
                    "status": "unreviewed",
                }
            )


if __name__ == "__main__":
    main()
