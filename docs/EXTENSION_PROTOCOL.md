# Extension protocol: citation fidelity and measurement validity

Fixed on 2026-10-14 before new audit/evaluation runs. Initial SciFact results have already been seen. This is a versioned plan, not a formal preregistration.

Question: which verifiable citation errors can a reproducible audit detect, and how much does automated grading depend on its own imperfect instrument?

1. Correct retrieval training to treat all annotated documents as positives. Separate document groups for fitting (60%), checkpoint selection (20%) and threshold calibration (20%); no reuse of checkpoint-selection claims for policy calibration. Record exact train/development claim overlap; report the official split and an explicitly labeled duplicate/near-text diagnostic subset.
2. Implement a system-independent report adapter. Validate claim/citation/quoted-passage/provenance records, hash source content, keep abstentions and access failures distinct, and test missing citations, source mismatches and unsupported automatic judgments.
3. Build a paired, controlled citation-corruption benchmark from expert-annotated SciFact cases. Keep each source claim as a dependence unit; original versus corrupted versions are not independent human annotations. Test changed source, altered quote, invalid sentence index and missing citation separately. Structural checks must not be described as semantic claim verification.
4. Prepare a deterministic blinded human-review packet and independent-review schema. With no genuine reviewer labels, report "unreviewed" and do not fabricate agreement or human-validated accuracy. Use automated NLI only as a proposal/instrument whose own accuracy is measured.
5. Inspect official published Kosmos reports/validation artifacts for a FREE retrospective adapter. Do not run a paid Kosmos/API experiment, consume unspecified credits, or treat our 46.7% SciFact NLI model as a gold judge of Kosmos. Publisher 79.4% statement accuracy and our citation metrics are different quantities.
6. Any deployed-system outcome requires actual accessible report inputs and reviewable judgments. If report/export access or original expert-label files are unavailable, record the limitation and publish the instrument/review workflow without claiming to have independently measured Kosmos accuracy.

Outstanding independent validation: external blinded annotators, agreement statistics, another annotated corpus, and accessible real-system outputs. A self-graded adapter alone does not satisfy these.

Implementation clarification: corpus-level loss retains the closed public index as negatives. Claim/reference groups are disjoint, but held-out document vectors can participate as negatives in fitting. The unseen-reference diagnostic is not unseen-document representation transfer. This clarification was added after the run; no result-based tuning was performed.
