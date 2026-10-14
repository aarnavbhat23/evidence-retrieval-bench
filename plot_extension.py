from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update(
    {
        "font.family": "DejaVu Sans",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.titlesize": 14,
        "font.size": 11,
        "figure.facecolor": "#fafbfd",
        "axes.facecolor": "#fafbfd",
    }
)
colors = {
    "additive": "#95a5ad",
    "matched_ridge": "#2d7790",
    "neural_ensemble": "#ba5b39",
    "endpoint_gaussian": "#95a5ad",
    "observation_likelihood": "#ba5b39",
    "constant_observation": "#2d7790",
}
p = Path("results/document-disjoint-v2")
report = json.loads((p / "metrics.json").read_text())
fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), layout="constrained")
methods = list(report["retrieval"])
x = np.arange(4)
for shift, name, label, color in [
    (-0.18, "official_annotated_dev", "Official 188 annotated claims", "#2d7790"),
    (
        0.18,
        "unseen_gold_document_and_no_exact_text_duplicate",
        "117 diagnostic claims",
        "#95a5ad",
    ),
]:
    axes[0].bar(
        x + shift,
        [report["retrieval"][m][name]["recall_at_5"] for m in methods],
        width=0.36,
        label=label,
        color=color,
    )
axes[0].set(
    xticks=x,
    xticklabels=["Lexical", "Frozen\nMiniLM", "Trained\nprojection", "Hybrid"],
    ylim=(0, 1),
    ylabel="Evidence recall @5",
    title="Retrieval",
)
axes[0].legend(fontsize=9, frameon=False)
rows = report["verification"]["development_risk_coverage"]
rows = [r for r in rows if r["accepted"]]
axes[1].plot(
    [r["coverage"] for r in rows],
    [r["label_and_citation_precision"] for r in rows],
    marker="o",
    color="#ba5b39",
)
axes[1].axhline(0.9, linestyle="--", color="#95a5ad", label="Desired precision")
axes[1].set(
    xlabel="Non-neutral coverage",
    ylabel="Correct label + cited evidence precision",
    ylim=(0, 1),
    title="Verifier failure persists",
)
axes[1].legend(frameon=False)
fig.suptitle(
    "Separate document-group calibration · all positives · no safety claim", fontsize=15
)
fig.savefig(p / "comparison.png", dpi=150)
fig.savefig(p / "comparison.svg")
plt.close(fig)
