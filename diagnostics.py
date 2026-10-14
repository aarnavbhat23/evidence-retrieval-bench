from pathlib import Path
import json,numpy as np,pandas as pd
from corpus import load_archive
from sklearn.feature_extraction.text import TfidfVectorizer

def main():
    root=Path(__file__).resolve().parent
    documents,train,dev=load_archive(root/'data/scifact.tar.gz');lookup={d['doc_id']:i for i,d in enumerate(documents)}
    pairs=[(i,lookup[int(sorted(c['evidence'],key=int)[0])]) for i,c in enumerate(train) if c['evidence']]
    a=np.array(pairs);unique=np.unique(a[:,1]);rng=np.random.default_rng(916);order=rng.permutation(unique);val=set(order[:max(1,len(order)//5)])
    ledger=[{'claim_id':train[i]['id'],'target_doc_id':documents[d]['doc_id'],'split':'validation' if d in val else 'train'} for i,d in pairs]
    pd.DataFrame(ledger).to_csv(root/'results/projection-split-ledger.csv',index=False)
    known={r['target_doc_id'] for r in ledger if r['split']=='train'}
    rankings=[json.loads(line) for line in (root/'results/retrieval-ledger.jsonl').read_text().splitlines()]
    report=json.loads((root/'results/metrics.json').read_text());strata={}
    for method in report['retrieval']:
     rows=[r for r in rankings if r['method']==method and r['gold_doc_ids']]
     strata[method]={}
     for label,subset in [('some_gold_doc_seen_in_projection_training',[r for r in rows if set(r['gold_doc_ids'])&known]),('no_gold_doc_seen_in_projection_training',[r for r in rows if not set(r['gold_doc_ids'])&known])]:
      strata[method][label]={'claims':len(subset),'recall_at_5':float(np.mean([bool(set(r['ranked_doc_ids'][:5])&set(r['gold_doc_ids'])) for r in subset]))}
    report['retrieval_training_document_strata']=strata
    vectorizer=TfidfVectorizer(ngram_range=(1,2),stop_words='english');matrix=vectorizer.fit_transform([c['claim'] for c in train+dev]);similarity=(matrix[len(train):]@matrix[:len(train)].T).toarray();best=similarity.argmax(1)
    pd.DataFrame([{'dev_claim_id':c['id'],'nearest_train_claim_id':train[j]['id'],'lexical_cosine':float(similarity[i,j])} for i,(c,j) in enumerate(zip(dev,best))]).to_csv(root/'results/train-dev-text-similarity.csv',index=False)
    report['claim_overlap_diagnostic']={'exact_train_dev_text_matches':len({c['claim'] for c in train}&{c['claim'] for c in dev}),'dev_claims_with_lexical_cosine_at_least_0_9':int((similarity.max(1)>=.9).sum()),'note':'Diagnostic, not deduplication. Official SciFact development set can contain related scientific propositions. This is not an independent new-paper test.'}
    (root/'results/metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps({'strata':strata['hybrid_rrf'],'overlap':report['claim_overlap_diagnostic']},indent=2))

if __name__ == "__main__": main()
