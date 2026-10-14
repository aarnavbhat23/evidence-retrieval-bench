"""Return ranked corpus citations for a supplied claim, with no generated answer."""
import argparse
import json
import numpy as np
import torch
from sklearn.feature_extraction.text import TfidfVectorizer
from transformers import AutoModel,AutoTokenizer
from corpus import MODEL,REVISION,load_archive,embed
from retrieval import Projection
from experiment import retrieve,passages


if __name__=='__main__':
    parser=argparse.ArgumentParser();parser.add_argument('claim');parser.add_argument('--device',default='cpu');args=parser.parse_args()
    documents,_,_=load_archive('data/scifact.tar.gz');vectors=np.load('data/embeddings.npz')['documents']
    tokenizer=AutoTokenizer.from_pretrained(MODEL,revision=REVISION)
    model=AutoModel.from_pretrained(MODEL,revision=REVISION).to(args.device).eval()
    query,_=embed([args.claim],tokenizer,model,args.device)
    projection=Projection().eval();projection.load_state_dict(torch.load('results/projection.pt',map_location='cpu',weights_only=True))
    vectorizer=TfidfVectorizer(ngram_range=(1,2),min_df=2,max_features=80000,stop_words='english')
    matrix=vectorizer.fit_transform([d['title']+' '+' '.join(d['abstract']) for d in documents])
    claim={'claim':args.claim};ranking=retrieve([claim],query,documents,vectors,vectorizer,matrix,projection)['hybrid_rrf'][0]
    retrieved=passages(claim,ranking,documents,vectorizer)
    print(json.dumps({'query':args.claim,'retrieved_passages':retrieved,
                      'note':'Retrieval matches are candidate evidence, not verified support. S2ORC abstracts retain ODC-By 1.0 attribution.'},indent=2))
