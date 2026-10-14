import json
from pathlib import Path
import numpy as np
from sklearn.metrics import accuracy_score,f1_score
from experiment import LABELS,selective_metrics


def main():
    report=json.loads(Path('results/metrics.json').read_text())
    import pandas as pd
    projection_ledger=pd.read_csv('results/projection-split-ledger.csv')
    assert projection_ledger.groupby('target_doc_id').split.nunique().max()==1
    ledger=[json.loads(line) for line in Path('results/retrieval-ledger.jsonl').read_text().splitlines()]
    for method,expected in report['retrieval'].items():
        rows=[r for r in ledger if r['method']==method and r['gold_doc_ids']]
        for k in [1,5,10,50]:
            value=np.mean([bool(set(r['ranked_doc_ids'][:k])&set(r['gold_doc_ids'])) for r in rows])
            assert np.isclose(value,expected[f'any_evidence_recall_at_{k}'])
    rows=[json.loads(line) for line in Path('results/verification.jsonl').read_text().splitlines()]
    assert len({r['claim_id'] for r in rows})==300
    expected=report['verification']
    assert np.isclose(accuracy_score([r['gold'] for r in rows],[r['prediction'] for r in rows]),expected['accuracy'])
    assert np.isclose(f1_score([r['gold'] for r in rows],[r['prediction'] for r in rows],labels=LABELS,average='macro'),expected['macro_f1'])
    for policy in expected['development_risk_coverage']:
        actual=selective_metrics(rows,policy['threshold'])
        for key,value in actual.items():
            assert value is None and policy[key] is None or value is not None and np.isclose(value,policy[key])
    print('Replayed four retrieval baselines, all 300 verdicts and every label-plus-citation precision point.')


if __name__=='__main__':main()
