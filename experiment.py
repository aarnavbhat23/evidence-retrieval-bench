"""Closed-corpus retrieval followed by non-oracle sentence-level NLI verification."""
import argparse
import copy
import json
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import accuracy_score,f1_score
from transformers import AutoModelForSequenceClassification,AutoTokenizer
from corpus import load_archive
from retrieval import Projection,contrastive_loss,rank,reciprocal_rank_fusion,gold_label

NLI='cross-encoder/nli-MiniLM2-L6-H768'
NLI_REVISION='b95119ce93d3e065de6214e38cd4a97b0f2f2c6d'
LABELS=['contradicted','supported','insufficient']


def train_projection(documents,claims,train_claims,lookup):
    pairs=[]
    for i,claim in enumerate(train_claims):
        if claim['evidence']:
            pairs.append((i,lookup[int(sorted(claim['evidence'],key=int)[0])]))
    pair=np.array(pairs);unique=np.unique(pair[:,1]);rng=np.random.default_rng(916)
    shuffled=rng.permutation(unique);val_docs=set(shuffled[:max(1,len(shuffled)//5)])
    val=np.array([d in val_docs for _,d in pairs]);training=pair[~val];validation=pair[val]
    torch.manual_seed(916);model=Projection();optimizer=torch.optim.AdamW(model.parameters(),lr=.001,weight_decay=.01)
    dq,cq=torch.tensor(documents),torch.tensor(claims)
    best,stale,state,history=float('inf'),0,None,[]
    for epoch in range(100):
        model.train();order=rng.permutation(len(training));losses=[]
        for offset in range(0,len(order),64):
            indexes=training[order[offset:offset+64]];optimizer.zero_grad()
            loss=contrastive_loss(model(cq[indexes[:,0]]),model(dq[indexes[:,1]]),torch.tensor(indexes[:,1]))
            loss.backward();optimizer.step();losses.append(float(loss.detach()))
        model.eval()
        with torch.no_grad():
            score=model(cq[validation[:,0]])@model(dq).T
            val_loss=float(torch.nn.functional.cross_entropy(score/.07,torch.tensor(validation[:,1])))
        history.append({'epoch':epoch+1,'train_contrastive':float(np.mean(losses)),'validation_corpus_ce':val_loss})
        if val_loss<best:best,stale,state=val_loss,0,copy.deepcopy(model.state_dict())
        else:stale+=1
        if stale==10:break
    model.load_state_dict(state);model.eval()
    return model,history,training,validation


def retrieve(claims,query_vectors,documents,document_vectors,vectorizer,matrix,projection):
    lexical=(vectorizer.transform([c['claim'] for c in claims])@matrix.T).toarray()
    dense=query_vectors@document_vectors.T
    with torch.no_grad():learned=(projection(torch.tensor(query_vectors))@projection(torch.tensor(document_vectors)).T).numpy()
    rankings={name:[] for name in ['tfidf','frozen_dense','trained_projection','hybrid_rrf']}
    for a,b,c in zip(lexical,dense,learned):
        ra,rb,rc=[rank(scores,50).tolist() for scores in [a,b,c]]
        for name,value in zip(rankings,[ra,rb,rc,reciprocal_rank_fusion([ra,rc])]):rankings[name].append(value)
    return rankings


def passages(claim,ranking,documents,vectorizer):
    items=[];query=vectorizer.transform([claim['claim']])
    for index in ranking[:3]:
        document=documents[index];sentences=document['abstract']
        if not sentences:continue
        scores=(query@vectorizer.transform(sentences).T).toarray()[0]
        selected=sorted(rank(scores,min(2,len(sentences))).tolist())
        text=' '.join(sentences[i] for i in selected)
        items.append({'doc_id':document['doc_id'],'sentence_indexes':selected,'text':text})
    return items


def verify(claims,rankings,documents,vectorizer,tokenizer,model,device):
    candidates=[passages(c,r,documents,vectorizer) for c,r in zip(claims,rankings)]
    pairs=[(candidate['text'],c['claim']) for c,items in zip(claims,candidates) for candidate in items]
    probabilities=[];truncated=0
    with torch.inference_mode():
        for offset in range(0,len(pairs),16):
            batch=pairs[offset:offset+16]
            truncated+=sum(len(tokenizer(a,b,truncation=False)['input_ids'])>512 for a,b in batch)
            tokens=tokenizer([p[0] for p in batch],[p[1] for p in batch],padding=True,truncation=True,max_length=512,return_tensors='pt')
            tokens={k:v.to(device) for k,v in tokens.items()}
            probabilities.extend(model(**tokens).logits.softmax(-1).cpu().numpy())
    records=[];offset=0
    for claim,items in zip(claims,candidates):
        values=np.array(probabilities[offset:offset+len(items)]);offset+=len(items)
        selected=int(np.argmax(1-values[:,2]));prob=values[selected];label=LABELS[int(prob.argmax())]
        candidate=items[selected]
        gold=gold_label(claim)
        records.append({'claim_id':claim['id'],'gold':gold,'prediction':label,'confidence':float(prob.max()),
            'doc_id':candidate['doc_id'],'sentence_indexes':candidate['sentence_indexes'],
            'probabilities':{k:float(v) for k,v in zip(LABELS,prob)},
            'citation_is_annotated_evidence':str(candidate['doc_id']) in claim['evidence'],
            'selected_sentences_are_gold_rationale':any(set(candidate['sentence_indexes']).issubset(set(e['sentences']))
                for e in claim['evidence'].get(str(candidate['doc_id']),[]))})
    return records,truncated


def selective_metrics(rows,threshold):
    accepted=[r for r in rows if r['prediction']!='insufficient' and r['confidence']>=threshold]
    return {'threshold':threshold,'accepted':len(accepted),'coverage':len(accepted)/len(rows),
            'label_precision':sum(r['prediction']==r['gold'] for r in accepted)/len(accepted) if accepted else None,
            'label_and_citation_precision':sum(r['prediction']==r['gold'] and r['citation_is_annotated_evidence'] for r in accepted)/len(accepted) if accepted else None}


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--data',type=Path,default=Path('data/scifact.tar.gz'))
    parser.add_argument('--device',default='mps' if torch.backends.mps.is_available() else 'cpu');args=parser.parse_args()
    torch.set_num_threads(4);documents,train,dev=load_archive(args.data);emb=np.load('data/embeddings.npz')
    lookup={d['doc_id']:i for i,d in enumerate(documents)};out=Path('results');out.mkdir(exist_ok=True)
    projection,history,training,validation=train_projection(emb['documents'],emb['train'],train,lookup)
    torch.save(projection.state_dict(),out/'projection.pt');(out/'history.json').write_text(json.dumps(history,indent=2)+'\n')
    text=[d['title']+' '+' '.join(d['abstract']) for d in documents]
    vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=80000,stop_words='english')
    matrix=vectorizer.fit_transform(text)
    rankings=retrieve(dev,emb['dev'],documents,emb['documents'],vectorizer,matrix,projection)
    report={'corpus_documents':len(documents),'train_claims':len(train),'dev_claims':len(dev),
        'projection_train_pairs':len(training),'projection_validation_pairs':len(validation),
        'selected_epoch':int(np.argmin([h['validation_corpus_ce'] for h in history]))+1,
        'retrieval':{},'encoder_note':'Frozen MiniLM may have seen scientific text in upstream pretraining; no contamination-free claim.'}
    annotated=[i for i,c in enumerate(dev) if c['evidence']];ledger=[]
    for name,lists in rankings.items():
        metrics={}
        for k in [1,5,10,50]:
            metrics[f'any_evidence_recall_at_{k}']=float(np.mean([bool(set(dev[i]['evidence']) & {str(documents[j]['doc_id']) for j in lists[i][:k]}) for i in annotated]))
        report['retrieval'][name]=metrics
        for claim,ranking in zip(dev,lists):
            ledger.append({'claim_id':claim['id'],'method':name,'ranked_doc_ids':[documents[i]['doc_id'] for i in ranking],
                           'gold_doc_ids':list(map(int,claim['evidence']))})
    tokenizer=AutoTokenizer.from_pretrained(NLI,revision=NLI_REVISION)
    model=AutoModelForSequenceClassification.from_pretrained(NLI,revision=NLI_REVISION).to(args.device).eval()
    results,truncated=verify(dev,rankings['hybrid_rrf'],documents,vectorizer,tokenizer,model,args.device)
    # Calibration claims use document groups withheld from projection fitting.
    calibration_claims=[train[i] for i in validation[:,0]]
    calibration_vectors=emb['train'][validation[:,0]]
    cr=retrieve(calibration_claims,calibration_vectors,documents,emb['documents'],vectorizer,matrix,projection)
    cal,_=verify(calibration_claims,cr['hybrid_rrf'],documents,vectorizer,tokenizer,model,args.device)
    thresholds=[0.,.5,.6,.7,.8,.9,.95,.99,1.]
    cal_metrics=[selective_metrics(cal,t) for t in thresholds]
    eligible=[m for m in cal_metrics if m['accepted']>=10 and m['label_and_citation_precision']>=.9]
    threshold=min(m['threshold'] for m in eligible) if eligible else 1.
    report['verification']={'nli_model':NLI,'revision':NLI_REVISION,'truncated_pairs':truncated,
        'accuracy':float(accuracy_score([r['gold'] for r in results],[r['prediction'] for r in results])),
        'macro_f1':float(f1_score([r['gold'] for r in results],[r['prediction'] for r in results],labels=LABELS,average='macro')),
        'development_risk_coverage':[selective_metrics(results,t) for t in thresholds],
        'calibration_risk_coverage':cal_metrics,'chosen_policy':selective_metrics(results,threshold),
        'policy_note':'Validation-selected empirical threshold, not a statistical safety guarantee. Labels and citations must both be correct.'}
    for name,values in [('retrieval-ledger',ledger),('verification',results),('calibration-verification',cal)]:
        (out/f'{name}.jsonl').write_text(''.join(json.dumps(v)+'\n' for v in values))
    (out/'metrics.json').write_text(json.dumps(report,indent=2)+'\n')
    from diagnostics import main as add_diagnostics
    add_diagnostics()
    print(json.dumps(report,indent=2))


if __name__=='__main__':main()
