import json
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

data=json.loads(Path('results/metrics.json').read_text());fig,axes=plt.subplots(1,2,figsize=(10,3.5),layout='constrained')
ks=[1,5,10,50]
for name,label in [('tfidf','Lexical TF-IDF'),('frozen_dense','Frozen MiniLM'),('trained_projection','Trained projection'),('hybrid_rrf','Hybrid fusion')]:
    axes[0].plot(ks,[data['retrieval'][name][f'any_evidence_recall_at_{k}'] for k in ks],'o-',label=label)
axes[0].set_xscale('log');axes[0].set_xticks(ks,ks);axes[0].set_xlabel('Retrieved abstracts');axes[0].set_ylabel('Any annotated evidence retrieved');axes[0].legend(fontsize=8,frameon=False)
policies=[p for p in data['verification']['development_risk_coverage'] if p['accepted']]
axes[1].plot([p['coverage'] for p in policies],[p['label_precision'] for p in policies],'o-',label='Correct verdict')
axes[1].plot([p['coverage'] for p in policies],[p['label_and_citation_precision'] for p in policies],'o-',label='Correct verdict and citation')
axes[1].axhline(.9,ls='--',color='#444',label='Desired precision');axes[1].set_ylim(0,1);axes[1].set_xlabel('Claims receiving a non-neutral verdict');axes[1].set_ylabel('Empirical precision');axes[1].legend(fontsize=8,frameon=False)
for ax in axes:ax.spines[['top','right']].set_visible(False)
fig.savefig('results/retrieval-verification.png',dpi=180);fig.savefig('results/retrieval-verification.svg')
