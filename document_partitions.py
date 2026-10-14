"""Partition related claims by every cited/evidence document, including neutral claims."""

import numpy as np


def references(claim):
    return sorted(
        set(map(int, claim["cited_doc_ids"])) | set(map(int, claim["evidence"]))
    )


def partition(claims, seed=916):
    parent = {d: d for c in claims for d in references(c)}

    def find(d):
        while parent[d] != d:
            parent[d] = parent[parent[d]]
            d = parent[d]
        return d

    for claim in claims:
        docs = references(claim)
        if not docs:
            raise ValueError(
                "A claim without document references needs a separate grouping rule"
            )
        for d in docs[1:]:
            parent[find(d)] = find(docs[0])
    components = {}
    for d in parent:
        components.setdefault(find(d), []).append(d)
    canonical = {
        d: tuple(sorted(members)) for members in components.values() for d in members
    }
    groups = sorted(set(canonical.values()))
    order = np.random.default_rng(seed).permutation(len(groups))
    assignment = {
        groups[g]: (
            "train"
            if i < 0.6 * len(groups)
            else "validation"
            if i < 0.8 * len(groups)
            else "calibration"
        )
        for i, g in enumerate(order)
    }
    labels = np.array([assignment[canonical[references(c)[0]]] for c in claims])
    docs_by_split = {
        s: {d for c, label in zip(claims, labels) if label == s for d in references(c)}
        for s in set(labels)
    }
    for a in docs_by_split:
        for b in docs_by_split:
            if a != b:
                assert not docs_by_split[a] & docs_by_split[b]
    return labels, groups
