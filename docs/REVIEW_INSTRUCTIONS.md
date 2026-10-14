# Blinded source review

No reviewer has completed this packet. `results/citation-challenge/blind-review-packet.jsonl` contains 60 unreviewed items. Do not interpret a generated packet or its controlled edits as human validation.

A coordinator should give the reviewer this document and the packet only. Withhold `cases.jsonl`, `review-construction-key.jsonl`, model verdicts and benchmark labels until judgments are locked. Reviewers should not inspect those public files while reviewing. Blinding here means withholding construction/model labels; visible corrupted quotations cannot be hidden.

For each claim, inspect its supplied sources, sentence indexes and quoted text. Use the full supplied abstract to avoid mistaking a selected sentence for the whole study. Judge support **from the supplied sources**, not from memory of whether the claim is generally true:

- `supports`: the source directly supports the claim, including population, intervention, comparison, outcome and direction.
- `refutes`: the source directly contradicts the claim. A merely irrelevant source does not refute it.
- `insufficient`: missing/irrelevant sources, overgeneralization, ambiguous statements or abstract-only evidence inadequate for the claim.

Retain `review_id`, claim, citations and sources. Set `reviewer_id` to a stable pseudonym, `judgment` to one of these values, `rationale` to a short explanation with source/sentence references, and `status` to `reviewed`. Optionally record `quote_integrity` (`matches`, `mismatch`, `not_checkable`) separately. Quote integrity does not establish scientific support. Leave uncertain items insufficient; do not force a verdict.

Each reviewer works independently. Freeze separate files before discussion. Then the coordinator can adjudicate disagreements, retaining original judgments and reasons. Do not give a reviewer the construction key as an answer key: it tracks edits and document identity, not semantic entailment. The packet includes paired edits of some of the same claims; those items are dependent. Report unique source claims alongside item counts.

Run `python review_packet.py path/to/review.jsonl --output review-summary.json`. It reports completion and actual inter-reviewer agreement when duplicate items have different reviewer IDs. It deliberately reports no accuracy without an independent adjudicated reference. Do not commit personal reviewer information; ask reviewers whether their pseudonym and comments may be published.
